import os
import random
from collections import OrderedDict
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


@dataclass
class Match:
    reference_patches: Optional[torch.Tensor] = None
    target_patches: Optional[torch.Tensor] = None

    patch_level_reference_coords: Optional[torch.Tensor] = None
    patch_level_target_coords: Optional[torch.Tensor] = None


def match_collate_fn(batch):
    reference_patches = torch.cat([match.reference_patches for match in batch], dim=0)
    target_patches = torch.cat([match.target_patches for match in batch], dim=0)

    patch_level_reference_coords = torch.cat([match.patch_level_reference_coords for match in batch], dim=0)
    patch_level_target_coords = torch.cat([match.patch_level_target_coords for match in batch], dim=0)

    return Match(
        reference_patches,
        target_patches,

        patch_level_reference_coords,
        patch_level_target_coords,
    )


class MatchesDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self,
                 stage, pair_names, patch_indices,
                 perturb_target=False,
                 patch_normalize=None,
                 image_augmentation_no_kp=None,
                 image_augmentation_kp=None):
        self.stage = stage
        self.perturb_target = perturb_target
        self.pair_names = pair_names
        self.patch_indices = patch_indices

        self.image_augmentation_no_kp = image_augmentation_no_kp
        self.image_augmentation_kp = image_augmentation_kp
        self.patch_normalize = patch_normalize

    def _setup_file(self):
        self.file = h5py.File(config.paths.matches, mode='r')
        self.references_group = self.file[f'{config.task.video}/{config.task.cam}/reference_coords']
        self.targets_group = self.file[f'{config.task.video}/{config.task.cam}/target_coords']

    def __len__(self):
        return len(self.pair_names)

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
                    x_min=left, y_min=upper,
                    x_max=right, y_max=lower,
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
        keypoint = int(keypoints[0][0]), int(keypoints[0][1])

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

            patches.append(patch)
            patch_level_coords.append(keypoint)

        reference_patches = torch.stack(patches)
        patch_level_reference_coords = torch.tensor(patch_level_coords, dtype=torch.int32)

        return reference_patches, patch_level_reference_coords

    def _prepare_targets(self, target_image_name, target_coords):
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

        import math
        padded_patch_size = desired_patch_size * 4

        patches = []
        patch_level_coords = []

        for index, (x, y) in enumerate(target_coords):
            keypoint = int(x), int(y)

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
                keypoint = int(x), int(y)

            center_point = x, y = keypoint

            if self.perturb_target:
                if self.stage in ['val', 'test']:
                    random.seed(index)

                perturb_size = (desired_patch_size - config.image.patch_border) // 2
                perturb_x = random.randint(-perturb_size, perturb_size)
                perturb_y = random.randint(-perturb_size, perturb_size)

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

            patches.append(patch)
            patch_level_coords.append(keypoint)

        target_patches = torch.stack(patches)
        patch_level_target_coords = torch.tensor(patch_level_coords, dtype=torch.int32)

        return target_patches, patch_level_target_coords

    def _prepare_images(self, idx):
        pair_name = self.pair_names[idx]
        patch_indices = self.patch_indices[pair_name]

        reference_coords = self.references_group[pair_name][()][patch_indices].astype(np.int32)
        target_coords = self.targets_group[pair_name][()][patch_indices].astype(np.int32)
        assert reference_coords.shape == target_coords.shape

        reference_image_name, target_image_name = pair_name.split('_')

        reference_coords = [(x, y) for x, y in reference_coords]
        references = self._prepare_references(
            reference_image_name,
            reference_coords
        )

        target_coords = [(x, y) for x, y in target_coords]
        targets = self._prepare_targets(
            target_image_name,
            target_coords
        )

        return references, targets

    def __getitem__(self, idx):
        if not hasattr(self, 'file'):
            self._setup_file()

        assert hasattr(self, 'file')

        references, targets = self._prepare_images(idx)
        reference_patches, patch_level_reference_coords = references
        target_patches, patch_level_target_coords = targets

        match = Match(
            reference_patches,
            target_patches,

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
                A.Defocus(p=0.3, radius=1),
                A.GaussNoise(p=0.3, var_limit=(20.0, 70.0)),
            ]
        )

        pad_mode = cv2.BORDER_CONSTANT  # cv2.BORDER_REPLICATE cv2.BORDER_CONSTANT
        pad_val = 0  # (0, 255, 0) if config.task.eda_mode else 0
        p = 1 if config.task.eda_mode else 0.4

        self.image_augmentation_kp = A.Compose(
            transforms=[
                A.Perspective(
                    # scale=(0.1, 0.3),
                    p=p,
                    fit_output=True,
                    pad_mode=pad_mode,
                    pad_val=pad_val,
                ),
                A.SafeRotate(
                    p=p,
                    limit=(-45, 45)
                ),
            ],
            keypoint_params=A.KeypointParams(format='xy')
        )

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            # T.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # ),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.dataset: Dict[str, MatchesDataset] = {}

        self.file = None
        self.references_group = None
        self.targets_group = None

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(config.paths.matches, mode='r')
            self.references_group = self.file[f'{config.task.video}/{config.task.cam}/reference_coords']
            self.targets_group = self.file[f'{config.task.video}/{config.task.cam}/target_coords']

    def _close_file(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.references_group = None
            self.targets_group = None

    def prepare_single_video_mode_data(self):
        patch_indices = OrderedDict()

        for a, b in zip(self.references_group.items(), self.targets_group.items()):
            (pair_name, ref_dataset), (_, tar_dataset) = a, b
            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            reference_coords = ref_dataset[()]
            target_coords = tar_dataset[()]

            reference_coords_len = len(reference_coords)
            target_coords_len = len(target_coords)
            assert reference_coords_len == target_coords_len

            N = min(reference_coords_len, config.train.num_patches_per_image)

            patch_indices_list = random.sample(range(reference_coords_len), N)
            patch_indices[pair_name] = patch_indices_list

        pair_names_list = list(self.references_group.keys())
        split_index = int(len(pair_names_list) * 0.8)

        pair_names = {
            'train': pair_names_list[:split_index],
            'test': pair_names_list[split_index:]
        }

        return pair_names, patch_indices

    def setup(self, stage=None):
        self._open_file()

        pair_names, patch_indices = None, None

        if config.task.single_video_mode:
            pair_names, patch_indices = self.prepare_single_video_mode_data()

        assert pair_names is not None, "pair_names is None"
        assert patch_indices is not None, "patch_indices is None"

        if stage == "fit":
            split_index = int(len(pair_names['train']) * 0.8)
            train_pair_names = pair_names['train'][:split_index]
            val_pair_names = pair_names['train'][split_index:]

            self.dataset['train'] = MatchesDataset(
                stage="train",
                pair_names=train_pair_names,
                patch_indices=patch_indices,
                perturb_target=True,  # True False

                patch_normalize=self.patch_normalize,
                image_augmentation_no_kp=self.image_augmentation_no_kp,
                image_augmentation_kp=self.image_augmentation_kp,
            )

            self.dataset['val'] = MatchesDataset(
                stage="val",
                pair_names=val_pair_names,
                patch_indices=patch_indices,
                perturb_target=True,

                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            self.dataset['test'] = MatchesDataset(
                stage="test",
                pair_names=pair_names['test'],
                patch_indices=patch_indices,
                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Test Dataset  : {len(self.dataset['test'])} samples")

    def teardown(self, stage=None):
        self._close_file()

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
