import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import albumentations as A
import h5py
import lightning as L
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    """
    Perform Min-Max Normalization on a tensor.
    Args:
        tensor (torch.Tensor): Input tensor with pixel values.
        min_val (float): Minimum value for normalization (default: 0.0).
        max_val (float): Maximum value for normalization (default: 1.0).
    Returns:
        torch.Tensor: Min-Max normalized tensor.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # Scale the tensor to the desired range
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    return normalized_tensor


@dataclass
class Match:
    reference_patches: Optional[torch.Tensor] = None
    target_patches: Optional[torch.Tensor] = None

    # rotations: Optional[torch.Tensor] = None

    patch_level_reference_coords: Optional[torch.Tensor] = None
    patch_level_target_coords: Optional[torch.Tensor] = None


def match_collate_fn(batch):
    reference_patches = torch.cat([match.reference_patches for match in batch], dim=0)
    target_patches = torch.cat([match.target_patches for match in batch], dim=0)

    # rotations = torch.cat([match.rotations for match in batch], dim=0)

    patch_level_reference_coords = torch.cat([match.patch_level_reference_coords for match in batch], dim=0)
    patch_level_target_coords = torch.cat([match.patch_level_target_coords for match in batch], dim=0)

    return Match(
        reference_patches,
        target_patches,

        # rotations,

        patch_level_reference_coords,
        patch_level_target_coords,
    )


class MatchesDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self,
                 stage, 
                 perturb_target=False,
                 patch_normalize=None,
                 image_augmentation_no_kp=None,
                 image_augmentation_kp=None):
        self.stage = stage
        self.perturb_target = perturb_target

        self.image_augmentation_no_kp = image_augmentation_no_kp
        self.image_augmentation_kp = image_augmentation_kp
        self.patch_normalize = patch_normalize

        self.names = []

    def _setup_file(self):
        self._file_all = h5py.File(config.paths.matches.all, mode='r')
        self._file_normal = h5py.File(config.paths.matches.normal, mode='r')
        self._file_blur= h5py.File(config.paths.matches.blur, mode='r')

        self.names = self._get_names()

    def _get_names(self):
        names = []
        ds_types = ['all', 'normal', 'blur']

        logger.info(f'Processing {self.stage}')

        for ds_type in ds_types:
            f = self._file_all 
            
            if ds_type == 'blur':
                f = self._file_blur
            elif ds_type == 'normal':
                f = self._file_normal

            logger.info(f'Processing {ds_type}')

            for video in config.task.videos[self.stage][ds_type]:
                config.task.video = video

                for cam in config.task.cams:
                    config.task.cam = cam

                    references = f[f'{video}/{cam}/reference_coords']
                    targets = f[f'{video}/{cam}/target_coords']

                    logger.info(f'Image Pair Counts : {video} {cam} : {len(references.items())}')

                    if len(references.items()) == 0:
                        continue

                    for a, b in zip(references.items(), targets.items()):
                        (pair_name, ref_dataset), (_, tar_dataset) = a, b
                        if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                            continue
                        
                        if len(ref_dataset[()]) == 0:
                            continue

                        names.append((ds_type, video, cam, pair_name))

        # print(names)

        return names

    def __len__(self):
        if not hasattr(self, '_file_all'):
            self._setup_file()

        assert hasattr(self, '_file_all')

        return len(self.names)

    @staticmethod
    def _get_image(image_name):
        image_path = os.path.join(config.paths.images, f'{image_name}.png')
        mode = 'RGB'
        image = Image.open(image_path).convert(mode)
        return image

    @staticmethod
    def _get_patch_boundary(image: Image.Image, center_point, patch_size):
        image_width, image_height = image.size
        x, y = center_point
        half_patch_size = patch_size // 2

        left, right = x - half_patch_size, x + half_patch_size
        upper, lower = y - half_patch_size, y + half_patch_size

        if left < 0:
            right += -left
            left = 0
        elif right > image_width:
            left -= (right - image_width)
            right = image_width

        if upper < 0:
            lower += -upper
            upper = 0
        elif lower > image_height:
            upper -= (lower - image_height)
            lower = image_height

        assert right > left
        assert right - left == patch_size
        assert lower > upper
        assert lower - upper == patch_size

        return left, upper, right, lower

    @staticmethod
    def _center_crop(image: Image.Image, keypoint, left, upper, right, lower):
        transform = A.Compose(
            transforms=[
                A.Crop(
                    x_min=round(left), y_min=round(upper),
                    x_max=round(right), y_max=round(lower),
                ),
            ],
            keypoint_params=A.KeypointParams(format='xy')
        )

        transformed = transform(image=np.array(image), keypoints=[keypoint])

        patch = Image.fromarray(transformed['image'])
        assert patch.size[0] == patch.size[1]
        assert patch.size[0] == right - left
        assert patch.size[1] == lower - upper

        keypoints = transformed['keypoints']
        assert len(keypoints) == 1, "Expected a single transformed keypoint"
        keypoint = keypoints[0][0], keypoints[0][1]

        return patch, keypoint

    def _prepare_references(self, reference_image_name, reference_coords):
        reference_image = self._get_image(reference_image_name)

        if self.image_augmentation_no_kp and self.stage == 'train':
            reference_image_np = np.array(reference_image)
            transformed = self.image_augmentation_no_kp(
                image=reference_image_np,
            )

            reference_image = Image.fromarray(transformed['image'])

        image_width, image_height = reference_image.size
        desired_patch_size = config.image.patch_size
        assert desired_patch_size < image_width
        assert desired_patch_size < image_height

        patches = []
        patch_level_coords = []

        for x, y in reference_coords:
            keypoint = x, y

            left, upper, right, lower = self._get_patch_boundary(
                reference_image,
                keypoint,
                desired_patch_size
            )

            patch, keypoint = self._center_crop(
                reference_image,
                keypoint,
                left, upper, right, lower
            )

            if self.patch_normalize:
                patch = self.patch_normalize(patch)
                patch = min_max_normalize(patch, min_val=0.0, max_val=1.0) 

            patches.append(patch)
            patch_level_coords.append(keypoint)

        reference_patches = torch.stack(patches)
        patch_level_reference_coords = torch.tensor(patch_level_coords, dtype=torch.float32)

        return reference_patches, patch_level_reference_coords

    def _prepare_targets(self, target_image_name, target_coords, image_idx):
        target_image = self._get_image(target_image_name)

        if self.stage == 'train' and self.image_augmentation_no_kp:
            target_image_np = np.array(target_image)
            transformed = self.image_augmentation_no_kp(
                image=target_image_np,
            )

            target_image = Image.fromarray(transformed['image'])

        image_width, image_height = target_image.size
        desired_patch_size = config.image.patch_size
        assert desired_patch_size < image_width
        assert desired_patch_size < image_height

        padded_patch_size = desired_patch_size * 4

        patches = []
        patch_level_coords = []

        for index, (x, y) in enumerate(target_coords):
            keypoint = x, y

            left, upper, right, lower = self._get_patch_boundary(
                target_image,
                keypoint,
                padded_patch_size
            )

            patch, keypoint = self._center_crop(
                target_image,
                keypoint,
                left, upper, right, lower
            )

            if self.stage == 'train' and self.image_augmentation_kp:
                patch_np = np.array(patch)
                transformed = self.image_augmentation_kp(
                    image=patch_np,
                    keypoints=[keypoint]
                )

                assert len(transformed['keypoints']) > 0, "No keypoints found after rotation"
                patch = Image.fromarray(transformed['image'])
                x, y = transformed['keypoints'][0]
                keypoint = x, y

            center_point = x, y = keypoint

            if self.perturb_target:
                if self.stage in ['val', 'test']:
                    np.random.seed(index + image_idx)

                perturb_size = (desired_patch_size - config.image.patch_border) // 2
                perturb_x = np.random.randint(2, perturb_size)
                perturb_y = np.random.randint(2, perturb_size)

                perturb_x = perturb_x * np.random.choice([1, -1])
                perturb_y = perturb_y * np.random.choice([1, -1])

                center_point = x + perturb_x, y + perturb_y

            left, upper, right, lower = self._get_patch_boundary(
                patch,
                center_point,
                desired_patch_size
            )

            patch, keypoint = self._center_crop(
                patch,
                keypoint,
                left, upper, right, lower
            )

            if self.patch_normalize:
                patch = self.patch_normalize(patch)
                patch = min_max_normalize(patch, min_val=0.0, max_val=1.0) 

            patches.append(patch)
            patch_level_coords.append(keypoint)

        target_patches = torch.stack(patches)
        patch_level_target_coords = torch.tensor(patch_level_coords, dtype=torch.float32)

        return target_patches, patch_level_target_coords

    def _prepare_images(self, idx):
        name = self.names[idx]
        ds_type, video, cam, pair_name = name
        reference_image_name, target_image_name = pair_name.split('_')

        config.task.video = video
        config.task.cam = cam

        f = self._file_all 
        
        if ds_type == 'blur':
            f = self._file_blur
        elif ds_type == 'normal':
            f = self._file_normal
            
        references_group = f[f'{config.task.video}/{config.task.cam}/reference_coords']
        targets_group = f[f'{config.task.video}/{config.task.cam}/target_coords']

        indices_group = f[f'{config.task.video}/{config.task.cam}/indices']
        indices = indices_group[pair_name][()].astype(np.int32)
        N = min(len(indices), config.train.num_patches_per_image)
        assert N != 0
        indices = indices[:N]

        reference_coords = references_group[pair_name][()][indices].astype(np.float32)
        target_coords = targets_group[pair_name][()][indices].astype(np.float32)
        assert reference_coords.shape == target_coords.shape

        reference_coords = [(x, y) for x, y in reference_coords]
        references = self._prepare_references(
            reference_image_name,
            reference_coords
        )

        target_coords = [(x, y) for x, y in target_coords]
        targets = self._prepare_targets(
            target_image_name,
            target_coords,
            idx
        )

        # rotations_group = f[f'{config.task.video}/{config.task.cam}/rotations']
        # rotations = rotations_group[pair_name][()][indices].astype(np.float32)
        # rotations = torch.tensor(rotations, dtype=torch.float32)

        return references, targets #, rotations

    def __getitem__(self, idx):
        if not hasattr(self, '_file_all'):
            self._setup_file()

        assert hasattr(self, '_file_all')

        # references, targets, rotations = self._prepare_images(idx)
        references, targets = self._prepare_images(idx)

        reference_patches, patch_level_reference_coords = references
        target_patches, patch_level_target_coords = targets

        match = Match(
            reference_patches,
            target_patches, 
            
            # rotations,

            patch_level_reference_coords,
            patch_level_target_coords,
        )

        return match


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.task.eda_mode else os.cpu_count()
        self.persistent_workers = not config.task.eda_mode

        self.image_augmentation_no_kp = A.Compose(
            transforms=[
                A.Defocus(p=0.5, radius=1),
                # A.GaussNoise(p=1, var_limit=0.001),
            ]
        )

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            # T.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # ),
        ])

        self.dataset: Dict[str, MatchesDataset] = {}
    
    def setup(self, stage=None):
        if stage == "fit":
            self.dataset['train'] = MatchesDataset(
                stage="train",
                perturb_target=True,  # True False

                patch_normalize=self.patch_normalize,
                image_augmentation_no_kp=self.image_augmentation_no_kp,
            )

            self.dataset['val'] = MatchesDataset(
                stage="val",
                perturb_target=True,
                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            self.dataset['test'] = MatchesDataset(
                stage="test",
                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Test Dataset  : {len(self.dataset['test'])} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['train'],
            batch_size=config.train.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            collate_fn=match_collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['val'],
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            collate_fn=match_collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['test'],
            batch_size=config.test.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            collate_fn=match_collate_fn
        )
