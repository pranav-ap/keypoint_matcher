from utils import logger
from config import config
import os
import torch
import random
import lightning as L
from PIL import Image
from typing import Optional
from torchvision import transforms as T


torch.set_float32_matmul_precision('medium')


def load_tensor(folder_path: str, filename: str) -> torch.tensor:
    filepath: str = os.path.join(folder_path, filename)
    assert os.path.exists(filepath), f'Tensor File Not Found : {filepath}'
    tensor: torch.tensor = torch.load(filepath, weights_only=True)
    return tensor


class MatchesDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self, images_dir, matches_dir, subset, transform=None, max_per_image=None, sample_size=None):
        self.subset = subset
        self.transform = transform
        self.max_per_image = max_per_image

        self.images_dir = images_dir
        self.matches_dir = matches_dir

        self.matches_names = sorted(os.listdir(self.matches_dir))

        if sample_size is not None:
            sample_indices = random.sample(range(len(self.matches_names)), min(sample_size, len(self.matches_names)))
            self.matches_names = [self.matches_names[i] for i in sample_indices]

    def __len__(self):
        return len(self.matches_names)

    def _get_items(self, idx):
        matches_filename = self.matches_names[idx]
        matches = load_tensor(self.matches_dir, matches_filename)

        left_coords = [(int(x), int(y)) for x, y in matches[:, :2]]
        right_coords = [(int(x), int(y)) for x, y in matches[:, 2:]]
        assert len(left_coords) == len(right_coords)

        if self.max_per_image is not None:
            sample_indices = random.sample(range(len(left_coords)), min(self.max_per_image, len(left_coords)))
            left_coords = [left_coords[i] for i in sample_indices]
            right_coords = [right_coords[i] for i in sample_indices]

        matches_filename, _ = matches_filename.split('.')
        reference_image_name, target_image_name, _ = matches_filename.split('_')

        reference_image_path = os.path.join(self.images_dir, reference_image_name)
        target_image_path = os.path.join(self.images_dir, target_image_name)

        reference_image = Image.open(reference_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        if self.transform:
            reference_image = self.transform(reference_image)
            target_image = self.transform(target_image)

        return left_coords, right_coords, reference_image, target_image

    @staticmethod
    def _get_patches(image, coords):
        patches = []

        width, height = config.image_size
        half_size = config.patch_size // 2

        for x, y in coords:
            # Calculate initial bounding box
            left = max(0, x - half_size)
            upper = max(0, y - half_size)
            right = min(width, x + half_size)
            lower = min(height, y + half_size)

            # Crop the patch from the image
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

        return patches

    def __getitem__(self, idx):
        left_coords, right_coords, reference_image, target_image = self._get_items(idx)

        reference_patches = self._get_patches(reference_image, left_coords)
        target_patches = self._get_patches(target_image, right_coords)
        assert len(reference_patches) == len(target_patches)

        return left_coords, right_coords, reference_patches, target_patches


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()

        w, h = config.image_size

        self.train_transform = T.Compose([
            T.Resize((w, h)),
            T.Grayscale(),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.test_transform = T.Compose([
            T.Resize((w, h)),
            T.Grayscale(),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.train_dataset: Optional[MatchesDataset] = None
        self.val_dataset: Optional[MatchesDataset] = None
        self.test_dataset: Optional[MatchesDataset] = None

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            images_dir = config.dirs.data
            matches_dir = config.dirs.data

            self.train_dataset = MatchesDataset(
                images_dir=images_dir,
                matches_dir=matches_dir,
                subset="train",
                transform=self.train_transform
            )

            images_dir = config.dirs.data
            matches_dir = config.dirs.data

            self.val_dataset = MatchesDataset(
                images_dir=images_dir,
                matches_dir=matches_dir,
                subset="validate",
                transform=self.test_transform
            )

            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == "test":
            images_dir = config.dirs.data
            matches_dir = config.dirs.data

            self.test_dataset = MatchesDataset(
                images_dir=images_dir,
                matches_dir=matches_dir,
                subset="test",
                transform=self.test_transform
            )

            logger.info(f"Test Dataset  : {len(self.test_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.test.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
