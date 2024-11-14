import math
import os
import random
from typing import Dict, Optional
from collections import OrderedDict

import h5py
import lightning as L
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
import albumentations as A

from config import config
from utils import logger
from dataclasses import dataclass

torch.set_float32_matmul_precision('medium')


@dataclass
class Match:
    reference_patches: Optional[torch.Tensor] = None
    target_patches: Optional[torch.Tensor] = None

    image_level_reference_coords: Optional[torch.Tensor] = None
    image_level_target_coords: Optional[torch.Tensor] = None

    patch_level_reference_coords: Optional[torch.Tensor] = None
    patch_level_target_coords: Optional[torch.Tensor] = None


def match_collate_fn(batch):
    channel = 3

    reference_patches = torch.stack([match.reference_patches for match in batch])
    reference_patches = reference_patches.reshape(-1, channel, config.image.patch_size, config.image.patch_size)

    target_patches = torch.stack([match.target_patches for match in batch])
    target_patches = target_patches.reshape(-1, channel, config.image.patch_size, config.image.patch_size)

    image_level_reference_coords = torch.stack([match.image_level_reference_coords for match in batch])
    image_level_reference_coords = image_level_reference_coords.reshape(-1, 2)

    image_level_target_coords = torch.stack([match.image_level_target_coords for match in batch])
    image_level_target_coords = image_level_target_coords.reshape(-1, 2)

    patch_level_reference_coords = torch.stack([match.patch_level_reference_coords for match in batch])
    patch_level_reference_coords = patch_level_reference_coords.reshape(-1, 2)

    patch_level_target_coords = torch.stack([match.patch_level_target_coords for match in batch])
    patch_level_target_coords = patch_level_target_coords.reshape(-1, 2)

    return Match(
        reference_patches,
        target_patches,

        image_level_reference_coords,
        image_level_target_coords,

        patch_level_reference_coords,
        patch_level_target_coords,
    )


class MatchesDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self, subset, pair_names, patch_indices, image_transform=None, patch_transform=None, draw_keypoint=False):
        self.subset = subset
        self.pair_names = pair_names
        self.image_transform = image_transform
        self.patch_transform = patch_transform

        self.patch_indices = patch_indices
        self.draw_keypoint = draw_keypoint

    def _setup_file(self):
        self.file = h5py.File(config.paths.matches, mode='r')
        self.references_group = self.file[f'{config.task.track}/{config.task.cam}/reference_coords']
        self.targets_group = self.file[f'{config.task.track}/{config.task.cam}/target_coords']

    def __len__(self):
        return len(self.pair_names)

    @staticmethod
    def _get_image(image_name):
        image_path = os.path.join(config.paths.images, f'{image_name}.png')
        image = Image.open(image_path).convert("RGB")
        return image

    def _get_image_level_items(self, idx, match: Match):
        pair_name = self.pair_names[idx]
        patch_indices = self.patch_indices[pair_name]

        reference_coords = self.references_group[pair_name][()][patch_indices]
        target_coords = self.targets_group[pair_name][()][patch_indices]
        assert reference_coords.shape == target_coords.shape

        match.image_level_reference_coords = torch.from_numpy(reference_coords)
        match.image_level_target_coords = torch.from_numpy(target_coords)

        reference_image_name, target_image_name = pair_name.split('_')

        reference_image = self._get_image(reference_image_name)
        target_image = self._get_image(target_image_name)

        if self.image_transform:
            reference_image = self.image_transform(reference_image)
            target_image = self.image_transform(target_image)

        return reference_image, target_image

    @staticmethod
    def _get_patch_boundary(image: Image.Image, keypoint, patch_size):
        image_width, image_height = image.size
        x, y = keypoint
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
        assert len(keypoints) == 1
        keypoint = int(keypoints[0][0]), int(keypoints[0][1])

        return patch, keypoint

    @staticmethod
    def _random_crop(patch: Image.Image, keypoint, desired_patch_size):
        width, height = patch.size
        x, y = keypoint
        half_patch_size = desired_patch_size // 2

        perturb_size = half_patch_size - 5
        perturb_x = random.randint(-perturb_size, perturb_size)
        perturb_y = random.randint(-perturb_size, perturb_size)

        center_x = x + perturb_x
        center_y = y + perturb_y

        left, right = center_x - half_patch_size, center_x + half_patch_size
        upper, lower = center_y - half_patch_size, center_y + half_patch_size

        if left < 0:
            right += -left
            left = 0
        elif right > width:
            left -= right - width
            right = width

        if upper < 0:
            lower += -upper
            upper = 0
        elif lower > height:
            upper -= lower - height
            lower = height

        transform = A.Compose(
            transforms=[
                A.Crop(
                    x_min=left, y_min=upper,
                    x_max=right, y_max=lower
                ),
            ],
            keypoint_params=A.KeypointParams(format='xy')
        )

        transformed = transform(image=np.array(patch), keypoints=[keypoint])

        patch = Image.fromarray(transformed['image'])
        assert patch.size[0] == patch.size[1]

        keypoints = transformed['keypoints']
        assert len(keypoints) == 1
        keypoint = int(keypoints[0][0]), int(keypoints[0][1])

        return patch, keypoint

    @staticmethod
    def _random_crop_old(patch, keypoint, desired_patch_size):
        transform = A.Compose(
            transforms=[
                A.RandomCrop(
                    width=desired_patch_size,
                    height=desired_patch_size
                ),
            ],
            keypoint_params=A.KeypointParams(format='xy')
        )

        transformed = transform(image=np.array(patch), keypoints=[keypoint])

        patch = Image.fromarray(transformed['image'])
        assert patch.size[0] == patch.size[1]

        keypoints = transformed['keypoints']
        assert len(keypoints) == 1
        keypoint = int(keypoints[0][0]), int(keypoints[0][1])

        return patch, keypoint

    def _prepare_patches(self, image: Image.Image, image_level_coords, perturb=False):
        patches = []
        patch_level_coords = []

        image_width, image_height = image.size
        desired_patch_size = config.image.patch_size
        # padded_patch_size = desired_patch_size + (desired_patch_size // 2)
        padded_patch_size = 2 * desired_patch_size - 2

        if not perturb:
            padded_patch_size = desired_patch_size

        assert desired_patch_size < image_width
        assert desired_patch_size < image_height
        assert padded_patch_size < image_width
        assert padded_patch_size < image_height

        for x, y in image_level_coords:
            keypoint = int(x.item()), int(y.item())

            left, upper, right, lower = self._get_patch_boundary(image, keypoint, padded_patch_size)
            patch, keypoint = self._center_crop(image, keypoint, left, upper, right, lower)

            if perturb:
                patch, keypoint = self._random_crop_old(patch, keypoint, desired_patch_size)

            if self.draw_keypoint:
                draw_im = ImageDraw.Draw(patch)
                radius = 2
                patch_level_x, patch_level_y = keypoint
                draw_im.ellipse((patch_level_x - radius, patch_level_y - radius, patch_level_x + radius, patch_level_y + radius), outline="red")

            if self.patch_transform:
                patch = self.patch_transform(patch)

            patches.append(patch)
            patch_level_coords.append(keypoint)

        patches = torch.stack(patches)
        patch_level_coords = torch.tensor(patch_level_coords, dtype=torch.int32)

        return patches, patch_level_coords

    def __getitem__(self, idx):
        if not hasattr(self, 'file'):
            self._setup_file()

        match = Match()
        reference_image, target_image = self._get_image_level_items(idx, match)

        patches, patch_level_coords = self._prepare_patches(reference_image, match.image_level_reference_coords)
        match.reference_patches, match.patch_level_reference_coords = patches, patch_level_coords

        patches, patch_level_coords = self._prepare_patches(target_image, match.image_level_target_coords, perturb=True)
        match.target_patches, match.patch_level_target_coords = patches, patch_level_coords

        return match


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.task.eda_mode else os.cpu_count()
        self.persistent_workers = not config.task.eda_mode
        self.draw_keypoint = config.task.eda_mode

        self.image_transform = T.Compose([])
        self.patch_transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.dataset: Dict[str, MatchesDataset] = {}
        self.patch_indices = OrderedDict()

        self.file = h5py.File(config.paths.matches, mode='r')
        self.references_group = self.file[f'{config.task.track}/{config.task.cam}/reference_coords']
        self.targets_group = self.file[f'{config.task.track}/{config.task.cam}/target_coords']

        self.pair_names = {}

    def prepare_data(self):
        if config.task.single_mode:
            self.prepare_single_mode_data()

    def prepare_single_mode_data(self):
        for (pair_name, ref_dataset), (_, tar_dataset) in zip(self.references_group.items(), self.targets_group.items()):
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
                subset="train",
                pair_names=train_pair_names,
                patch_indices=self.patch_indices,
                patch_transform=self.patch_transform,
                draw_keypoint=self.draw_keypoint
            )

            self.dataset['val'] = MatchesDataset(
                subset="validate",
                pair_names=val_pair_names,
                patch_indices=self.patch_indices,
                patch_transform=self.patch_transform,
                draw_keypoint=self.draw_keypoint
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            self.dataset['test'] = MatchesDataset(
                subset="train",
                pair_names=self.pair_names['test'],
                patch_indices=self.patch_indices,
                patch_transform=self.patch_transform,
                draw_keypoint=self.draw_keypoint
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
