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
    channel = 3

    reference_patches = torch.stack([match.reference_patches for match in batch])
    reference_patches = reference_patches.reshape(-1, channel, config.image.patch_size, config.image.patch_size)

    target_patches = torch.stack([match.target_patches for match in batch])
    target_patches = target_patches.reshape(-1, channel, config.image.patch_size, config.image.patch_size)

    patch_level_reference_coords = torch.stack([match.patch_level_reference_coords for match in batch])
    patch_level_reference_coords = patch_level_reference_coords.reshape(-1, 2)

    patch_level_target_coords = torch.stack([match.patch_level_target_coords for match in batch])
    patch_level_target_coords = patch_level_target_coords.reshape(-1, 2)

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
                 patch_normalize=None,
                 patch_augmentation_no_kp=None,
                 patch_augmentation_kp=None):
        self.stage = stage
        self.pair_names = pair_names
        self.patch_indices = patch_indices

        self.patch_normalize = patch_normalize
        self.patch_augmentation_no_kp = patch_augmentation_no_kp
        self.patch_augmentation_kp = patch_augmentation_kp

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

    def _prepare_images(self, idx):
        pair_name = self.pair_names[idx]
        patch_indices = self.patch_indices[pair_name]

        reference_coords = self.references_group[pair_name][()][patch_indices]
        target_coords = self.targets_group[pair_name][()][patch_indices]
        assert reference_coords.shape == target_coords.shape

        image_level_reference_coords = torch.tensor(reference_coords, dtype=torch.int32)
        image_level_target_coords = torch.tensor(target_coords, dtype=torch.int32)

        reference_image_name, target_image_name = pair_name.split('_')

        reference_image = self._get_image(reference_image_name)
        target_image = self._get_image(target_image_name)

        return reference_image, target_image, image_level_reference_coords, image_level_target_coords

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
                    x_max=right, y_max=lower
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

    def _prepare_reference_patches(self, image: Image.Image, image_level_coords, match: Match):
        patches = []
        patch_level_coords = []

        image_width, image_height = image.size

        desired_patch_size = config.image.patch_size
        assert desired_patch_size < image_width
        assert desired_patch_size < image_height

        for x, y in image_level_coords:
            keypoint = int(x.item()), int(y.item())

            left, upper, right, lower = self._get_patch_boundary(image, keypoint, desired_patch_size)
            patch, keypoint = self._center_crop(image, keypoint, left, upper, right, lower)

            if self.patch_augmentation_no_kp:
                patch_np = np.array(patch)
                transformed = self.patch_augmentation_no_kp(
                    image=patch_np,
                )

                patch = Image.fromarray(transformed['image'])

            if self.patch_normalize:
                patch = self.patch_normalize(patch)

            patches.append(patch)
            patch_level_coords.append(keypoint)

        match.reference_patches = torch.stack(patches)
        match.patch_level_reference_coords = torch.tensor(patch_level_coords, dtype=torch.int32)

    def _prepare_target_patches(self, image: Image.Image, image_level_coords, match: Match):
        patches = []
        patch_level_coords = []

        image_width, image_height = image.size

        desired_patch_size = config.image.patch_size
        assert desired_patch_size < image_width
        assert desired_patch_size < image_height

        padded_patch_size = 2 * (desired_patch_size - config.image.patch_border)
        assert desired_patch_size < padded_patch_size

        for index, (x, y) in enumerate(image_level_coords):
            keypoint = int(x.item()), int(y.item())

            left, upper, right, lower = self._get_patch_boundary(image, keypoint, padded_patch_size)
            patch, keypoint = self._center_crop(image, keypoint, left, upper, right, lower)

            if self.patch_augmentation_no_kp:
                patch_np = np.array(patch)
                transformed = self.patch_augmentation_no_kp(
                    image=patch_np
                )

                patch = Image.fromarray(transformed['image'])

            if self.patch_augmentation_kp and self.stage == 'train':
                patch_np = np.array(patch)
                transformed = self.patch_augmentation_kp(
                    image=patch_np,
                    keypoints=[keypoint]
                )

                patch = Image.fromarray(transformed['image'])
                keypoints = transformed['keypoints']

                assert len(keypoints) == 1, "Expected a single transformed keypoint"
                keypoint = (int(keypoints[0][0]), int(keypoints[0][1]))

            x, y = center_point = keypoint

            if self.stage in ['val', 'test']:
                random.seed(index)

            perturb_size = (desired_patch_size - config.image.patch_border) // 2
            perturb_x = random.randint(-perturb_size, perturb_size)
            perturb_y = random.randint(-perturb_size, perturb_size)

            center_point_x, center_point_y = x + perturb_x, y + perturb_y
            center_point_x = max(0, min(center_point_x, padded_patch_size - 1))
            center_point_y = max(0, min(center_point_y, padded_patch_size - 1))

            center_point = center_point_x, center_point_y

            left, upper, right, lower = self._get_patch_boundary(patch, center_point, desired_patch_size)
            patch, keypoint = self._center_crop(patch, keypoint, left, upper, right, lower)

            if self.patch_normalize:
                patch = self.patch_normalize(patch)

            patches.append(patch)
            patch_level_coords.append(keypoint)

        match.target_patches = torch.stack(patches)
        match.patch_level_target_coords = torch.tensor(patch_level_coords, dtype=torch.int32)

    def __getitem__(self, idx):
        if not hasattr(self, 'file'):
            self._setup_file()

        assert hasattr(self, 'file')

        match = Match()

        package = self._prepare_images(idx)
        reference_image, target_image, image_level_reference_coords, image_level_target_coords = package

        self._prepare_reference_patches(
            reference_image,
            image_level_reference_coords,
            match
        )

        self._prepare_target_patches(
            target_image,
            image_level_target_coords,
            match
        )

        return match


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.task.eda_mode else os.cpu_count()
        self.persistent_workers = not config.task.eda_mode

        self.patch_augmentation_no_kp = A.Compose(
            transforms=[
                A.Defocus(p=0.3, radius=2),
            ]
        )

        always_apply = config.task.eda_mode
        pad_mode = cv2.BORDER_CONSTANT  # cv2.BORDER_REPLICATE
        pad_val = (0, 255, 0) if config.task.eda_mode else 0

        self.patch_augmentation_kp = A.Compose(
            transforms=[
                A.Rotate(p=0.6, always_apply=always_apply),
                A.Perspective(fit_output=True, pad_mode=pad_mode, pad_val=pad_val, p=0.6, always_apply=always_apply),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
            ],
            keypoint_params=A.KeypointParams(format='xy')
        )

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.dataset: Dict[str, MatchesDataset] = {}
        self.patch_indices = OrderedDict()

        self.file = h5py.File(config.paths.matches, mode='r')
        self.references_group = self.file[f'{config.task.video}/{config.task.cam}/reference_coords']
        self.targets_group = self.file[f'{config.task.video}/{config.task.cam}/target_coords']

        self.pair_names = {}

    def prepare_data(self):
        if config.task.single_video_mode:
            self.prepare_single_video_mode_data()

    def prepare_single_video_mode_data(self):
        for a, b in zip(self.references_group.items(), self.targets_group.items()):
            (pair_name, ref_dataset), (_, tar_dataset) = a, b
            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            reference_coords = ref_dataset[()]

            reference_coords_len = len(reference_coords)
            N = min(reference_coords_len, config.train.num_patches_per_image)

            patch_indices = random.sample(range(reference_coords_len), N)
            self.patch_indices[pair_name] = patch_indices

        pair_names = list(self.references_group.keys())
        split_index = int(len(pair_names) * 0.8)
        self.pair_names['train'] = pair_names[:split_index]
        self.pair_names['test'] = pair_names[split_index:]

    def setup(self, stage=None):
        if stage == "fit":
            split_index = int(len(self.pair_names['train']) * 0.8)
            train_pair_names = self.pair_names['train'][:split_index]
            val_pair_names = self.pair_names['train'][split_index:]

            self.dataset['train'] = MatchesDataset(
                stage="train",
                pair_names=train_pair_names,
                patch_indices=self.patch_indices,

                patch_normalize=self.patch_normalize,
                patch_augmentation_no_kp=self.patch_augmentation_no_kp,
                patch_augmentation_kp=self.patch_augmentation_kp,
            )

            self.dataset['val'] = MatchesDataset(
                stage="val",
                pair_names=val_pair_names,
                patch_indices=self.patch_indices,

                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            self.dataset['test'] = MatchesDataset(
                stage="train",
                pair_names=self.pair_names['test'],
                patch_indices=self.patch_indices,
                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Test Dataset  : {len(self.dataset['test'])} samples")

    def teardown(self, stage=None):
        if self.file:
            self.file.close()

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
