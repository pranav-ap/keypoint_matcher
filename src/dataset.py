import os
import random
from typing import Optional
from collections import OrderedDict

import h5py
import lightning as L
import torch
from PIL import Image
from torchvision import transforms as T

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


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

    def _get_items(self, idx):
        pair_name = self.pair_names[idx]
        selected_indices = self.selected_indices[pair_name]

        reference_coords = self.references_group[pair_name][()][selected_indices]
        target_coords = self.targets_group[pair_name][()][selected_indices]

        logger.info(reference_coords)

        assert len(reference_coords) == len(target_coords)

        reference_image_name, target_image_name = pair_name.split('_')

        reference_image = self._get_image(reference_image_name)
        target_image = self._get_image(target_image_name)

        if self.image_transform:
            reference_image = self.image_transform(reference_image)
            target_image = self.image_transform(target_image)

        return reference_coords, target_coords, reference_image, target_image

    def _get_patches(self, image, coords):
        patches = []

        width, height = config.image.original_image_shape
        half_size = config.image.patch_size // 2

        for x, y in coords:
            # Calculate initial bounding box
            left = max(0, x - half_size)
            upper = max(0, y - half_size)
            right = min(width, x + half_size)
            lower = min(height, y + half_size)

            # Crop the patch from the image
            patch = image.crop((left, upper, right, lower))

            if self.patch_transform:
                patch = self.patch_transform(patch)

            patches.append(patch)

        return patches

    def __getitem__(self, idx):
        if not hasattr(self, 'file'):
            self._setup_file()

        reference_coords, target_coords, reference_image, target_image = self._get_items(idx)

        reference_patches = self._get_patches(reference_image, reference_coords)
        target_patches = self._get_patches(target_image, target_coords)
        assert len(reference_patches) == len(target_patches)

        return reference_coords, target_coords, reference_patches, target_patches


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

        self.train_dataset: Optional[MatchesDataset] = None
        self.val_dataset: Optional[MatchesDataset] = None
        self.test_dataset: Optional[MatchesDataset] = None

        self.selected_indices = OrderedDict()

        self.file = h5py.File(config.paths.matches, mode='r')
        self.references_group = self.file[f'{config.task.track}/{config.task.cam}/reference_coords']
        self.targets_group = self.file[f'{config.task.track}/{config.task.cam}/target_coords']

    def prepare_data(self):
        if config.task.single_mode:
            self.prepare_single_mode_data()

    def prepare_single_mode_data(self):
        for (pair_name, ref_dataset), (_, tar_dataset) in zip(self.references_group.items(), self.targets_group.items()):
            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            reference_coords = ref_dataset[()]

            reference_coords_len = len(reference_coords)
            N = min(config.train.num_patches_per_image, reference_coords_len)

            selected_indices = random.sample(range(reference_coords_len), N)
            self.selected_indices[pair_name] = selected_indices

    def setup(self, stage=None):
        if stage == "fit":
            pair_names = list(self.references_group.keys())
            split_index = int(len(pair_names) * 0.7)
            train_pair_names, val_pair_names = pair_names[:split_index], pair_names[split_index:]

            self.file.close()

            self.train_dataset = MatchesDataset(
                subset="train",
                pair_names=train_pair_names,
                selected_indices=self.selected_indices,
                patch_transform=self.patch_transform
            )

            self.val_dataset = MatchesDataset(
                subset="validate",
                pair_names=val_pair_names,
                selected_indices=self.selected_indices,
                patch_transform=self.patch_transform
            )

            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

    def teardown(self, stage=None):
        if self.file:
            self.file.close()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
