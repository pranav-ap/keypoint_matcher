import os
import random
from typing import Dict, Optional
from collections import OrderedDict

import h5py
import lightning as L
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

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

    # patch_level_reference_coords: Optional[torch.Tensor | np.array] = None
    # patch_level_target_coords: Optional[torch.Tensor | np.array] = None


def match_collate_fn(batch):
    reference_patches = torch.stack([match.reference_patches for match in batch])
    target_patches = torch.stack([match.target_patches for match in batch])
    image_level_reference_coords = torch.stack([match.image_level_reference_coords for match in batch])
    image_level_target_coords = torch.stack([match.image_level_target_coords for match in batch])

    reference_patches = reference_patches.reshape(-1, config.image.patch_size, config.image.patch_size)
    target_patches = target_patches.reshape(-1, config.image.patch_size, config.image.patch_size)
    image_level_reference_coords = image_level_reference_coords.reshape(-1, 2)
    image_level_target_coords = image_level_target_coords.reshape(-1, 2)

    return Match(
        reference_patches=reference_patches,
        target_patches=target_patches,

        image_level_reference_coords=image_level_reference_coords,
        image_level_target_coords=image_level_target_coords,
    )


class MatchesDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self, subset, pair_names, selected_indices, image_transform=None, patch_transform=None):
        self.subset = subset
        self.pair_names = pair_names
        self.image_transform = image_transform
        self.patch_transform = patch_transform

        self.selected_indices = selected_indices

    def _setup_file(self):
        self.file = h5py.File(config.paths.matches, mode='r')
        self.references_group = self.file[f'{config.task.track}/{config.task.cam}/reference_coords']
        self.targets_group = self.file[f'{config.task.track}/{config.task.cam}/target_coords']

    def __len__(self):
        return len(self.pair_names)

    @staticmethod
    def _get_image(image_name):
        image_path = os.path.join(config.paths.images, f'{image_name}.png')
        image = Image.open(image_path).convert("L")
        return image

    def _get_items(self, idx, match: Match):
        pair_name = self.pair_names[idx]
        selected_indices = self.selected_indices[pair_name]

        match.image_level_reference_coords = self.references_group[pair_name][()][selected_indices]
        match.image_level_target_coords = self.targets_group[pair_name][()][selected_indices]
        assert len(match.image_level_reference_coords) == len(match.image_level_target_coords)

        reference_image_name, target_image_name = pair_name.split('_')

        reference_image = self._get_image(reference_image_name)
        target_image = self._get_image(target_image_name)

        if self.image_transform:
            reference_image = self.image_transform(reference_image)
            target_image = self.image_transform(target_image)

        return reference_image, target_image

    @staticmethod
    def _get_patch_dims(x, y):
        width, height = config.image.original_image_shape
        half_size = config.image.patch_size // 2

        left = x - half_size
        upper = y - half_size
        right = x + half_size
        lower = y + half_size

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

        return left, upper, right, lower

    @staticmethod
    def _perturb(x, y, left, upper, right, lower):
        width, height = config.image.original_image_shape
        half_size = config.image.patch_size // 2

        perturb_size = half_size - 5
        perturb_x = random.randint(-perturb_size, perturb_size)
        perturb_y = random.randint(-perturb_size, perturb_size)

        x += perturb_x
        y += perturb_y

        left += perturb_x
        y += perturb_y
        x += perturb_x
        y += perturb_y

    def _prepare_patches(self, image, coords, perturb=False, draw=False):
        patches = []

        for x, y in coords:
            left, upper, right, lower = self._get_patch_dims(x, y)

            if perturb:
                self._perturb(x, y, left, upper, right, lower)

            patch = image.crop((left, upper, right, lower))

            if self.patch_transform:
                patch = self.patch_transform(patch)
                patch = patch.squeeze()

            patches.append(patch)

        patches = torch.stack(patches)

        return patches

    def __getitem__(self, idx):
        if not hasattr(self, 'file'):
            self._setup_file()

        match = Match()
        reference_image, target_image = self._get_items(idx, match)

        match.reference_patches = self._prepare_patches(reference_image, match.image_level_reference_coords)
        match.target_patches = self._prepare_patches(target_image, match.image_level_target_coords)

        match.image_level_reference_coords = torch.from_numpy(match.image_level_reference_coords)
        match.image_level_target_coords = torch.from_numpy(match.image_level_target_coords)

        return match


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()

        self.image_transform = T.Compose([])
        self.patch_transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.dataset: Dict[str, MatchesDataset] = {}
        self.selected_indices = OrderedDict()

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

            selected_indices = random.sample(range(reference_coords_len), N)
            self.selected_indices[pair_name] = selected_indices

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
                selected_indices=self.selected_indices,
                patch_transform=self.patch_transform
            )

            self.dataset['val'] = MatchesDataset(
                subset="validate",
                pair_names=val_pair_names,
                selected_indices=self.selected_indices,
                patch_transform=self.patch_transform
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            self.dataset['test'] = MatchesDataset(
                subset="train",
                pair_names=self.pair_names['test'],
                selected_indices=self.selected_indices,
                patch_transform=self.patch_transform
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
            persistent_workers=True,
            pin_memory=True,
            collate_fn=match_collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['val'],
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=match_collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['test'],
            batch_size=config.test.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=match_collate_fn
        )

