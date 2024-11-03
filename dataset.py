from utils import logger
from config import config
import os
import torch
import random
import lightning as L
from PIL import Image, ImageDraw
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
    def __init__(self, images_dir, matches_dir, matches_filenames, max_per_image=None):
        self.max_per_image = max_per_image

        self.images_dir = images_dir
        self.matches_dir = matches_dir
        self.matches_filenames = matches_filenames

        w, h = config.image_size
        self.transform_pil = T.Compose([
            T.Resize((w, h)),
            T.Grayscale(),
        ])
        
        self.transform_tensor = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.matches_filenames)

    def _get_coords(self, idx):
        matches_filename = self.matches_filenames[idx]
        matches = load_tensor(self.matches_dir, matches_filename)

        left_coords = [(int(x), int(y)) for x, y in matches[:, :2]]
        right_coords = [(int(x), int(y)) for x, y in matches[:, 2:]]
        assert len(left_coords) == len(right_coords)
        
        if self.max_per_image is not None:
            sample_indices = random.sample(range(len(left_coords)), min(self.max_per_image, len(left_coords)))
            left_coords = [left_coords[i] for i in sample_indices]
            right_coords = [right_coords[i] for i in sample_indices]

        return left_coords, right_coords

    def _get_images(self, idx):
        matches_filename = self.matches_filenames[idx]
        matches_filename, _ = matches_filename.split('.')
        reference_image_name, target_image_name, _ = matches_filename.split('_')

        reference_image_path = os.path.join(self.images_dir, f'{reference_image_name}.png')
        assert os.path.exists(reference_image_path)
        target_image_path = os.path.join(self.images_dir, f'{target_image_name}.png')
        assert os.path.exists(target_image_path)
        
        reference_image = Image.open(reference_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        reference_image = self.transform_pil(reference_image) 
        target_image = self.transform_pil(target_image) 
        
        return reference_image, target_image
        
    def _get_patches(self, image, coords, perturb=False):
        patches, adjusted_coords = [], []
        patch_size = config.patch_size
        half_size = patch_size // 2
    
        for x, y in coords:
            original_x, original_y = x, y
    
            if perturb:
                perturb_x = random.randint(-half_size // 2, half_size // 2)
                perturb_y = random.randint(-half_size // 2, half_size // 2)
                x += perturb_x
                y += perturb_y
    
            x = min(max(x, half_size), image.width - half_size - 1)
            y = min(max(y, half_size), image.height - half_size - 1)
    
            left = x - half_size
            upper = y - half_size
    
            if left < 0:
                left = 0
            elif left + patch_size > image.width:
                left = image.width - patch_size - 5
            
            if upper < 0:
                upper = 0
            elif upper + patch_size > image.height:
                upper = image.height - patch_size - 5
    
            right = left + patch_size
            lower = upper + patch_size

            patch = image.crop((left, upper, right, lower))
    
            rel_x = min(max(original_x - left, 0), patch_size - 1)
            rel_y = min(max(original_y - upper, 0), patch_size - 1)            
            adjusted_coords.append((rel_x, rel_y))
    
            draw = ImageDraw.Draw(patch)
            radius = 3
            draw.ellipse((rel_x - radius, rel_y - radius, rel_x + radius, rel_y + radius), outline="red")
    
            patch_tensor = self.transform_tensor(patch)
            patches.append(patch_tensor)
    
        patches = torch.stack(patches) # .squeeze()
        adjusted_coords = torch.tensor(adjusted_coords)
    
        return patches, adjusted_coords

    def __getitem__(self, idx):
        left_coords, right_coords = self._get_coords(idx)        
        reference_image, target_image = self._get_images(idx)
                       
        reference_patches, left_coordsp = self._get_patches(reference_image, left_coords)
        target_patches, right_coordsp = self._get_patches(target_image, right_coords, perturb=True)
        assert reference_patches.shape == target_patches.shape
        
        return left_coordsp, right_coordsp, reference_patches, target_patches


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()

        self.train_dataset: Optional[MatchesDataset] = None
        self.val_dataset: Optional[MatchesDataset] = None
        self.test_dataset: Optional[MatchesDataset] = None

    def setup(self, stage=None):        
        if stage == "fit" or stage == "validate":
            matches_filenames = sorted(os.listdir(config.train.matches))
            matches_filenames = random.sample(matches_filenames, 50)
            
            self.train_dataset = MatchesDataset(
                images_dir=config.train.images,
                matches_dir=config.train.matches,
                matches_filenames=matches_filenames[:30],
                max_per_image=config.train.max_per_image
            )

            self.val_dataset = MatchesDataset(
                images_dir=config.train.images,
                matches_dir=config.train.matches,
                matches_filenames=matches_filenames[30:],
                max_per_image=config.train.max_per_image
            )

            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == "test":
            matches_filenames = sorted(os.listdir(config.test.matches))
            matches_filenames = random.sample(matches_filenames, 20)
            
            self.test_dataset = MatchesDataset(
                images_dir=config.test.images,
                matches_dir=config.test.matches,
                matches_filenames=matches_filenames,
                max_per_image=config.test.max_per_image
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
