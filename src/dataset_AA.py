from config import config
from utils import logger, min_max_normalize

import os
import gc
from typing import Dict, Optional
import numpy as np
import pandas as pd
import cv2
import albumentations as A
import h5py
import lightning as L
import torch
from PIL import Image
from torchvision import transforms as T
import math
import random


def match_collate_fn(batch):
    ref_patches, references, tar_patches, targets, estimates, certainties = zip(*batch)

    # Convert keypoints from list of tuples to tensor
    references = torch.tensor(references, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    certainties = torch.tensor(certainties, dtype=torch.float32)
    estimates = torch.tensor(estimates, dtype=torch.float32)

    return (
        torch.cat(ref_patches, dim=0).unsqueeze(1),
        references,
        torch.cat(tar_patches, dim=0).unsqueeze(1),
        targets,
        estimates,
        certainties,
    )


def get_patch_boundary(image: Image.Image, center_point, patch_size):
    image_width, image_height = image.size
    x, y = center_point
    half_patch_size = patch_size // 2

    # Validate inputs
    assert patch_size > 0, "Patch size must be positive."
    assert patch_size <= image_width and patch_size <= image_height, "Patch size must be smaller than or equal to image dimensions."

    # Round center point to integers
    x, y = int(round(x)), int(round(y))

    # Calculate patch boundaries
    left = max(0, min(math.floor(x - half_patch_size), image_width - patch_size))
    upper = max(0, min(math.floor(y - half_patch_size), image_height - patch_size))
    right, lower = left + patch_size, upper + patch_size

    # Log warnings for edge cases
    # if (
    #     x - half_patch_size < 0 or 
    #     x + half_patch_size > image_width or 
    #     y - half_patch_size < 0 or 
    #     y + half_patch_size > image_height
    # ):
    #     logger.warn(f"Patch with {x, y=} is not fully centered due to image boundaries.")

    # Validate patch dimensions
    assert right > left, f"Left: {left}, Right: {right}"
    assert right - left == patch_size, f"Right - Left: {right - left}"
    assert lower > upper, f"Upper: {upper}, Lower: {lower}"
    assert lower - upper == patch_size, f"Lower - Upper: {lower - upper}"

    return left, upper, right, lower


def random_symmetric(x):
    x = min(config.image.max_perturb, x) 
    return random.uniform(-x, x)


def prepare_jitter_reference_patch(image, old_keypoint, patch_size=128, stage='train'):
    x0, y0 = old_keypoint
    
    perturb_x, perturb_y = 0, 0
    padding = config.image.patch_padding 

    assert patch_size > 0, "Patch size must be positive."
    assert patch_size <= image.width and patch_size <= image.height, "Patch size must be smaller than or equal to image dimensions."

    if config.image.jitter_reference and stage == 'train':
        max_perturb_x = max(0, min(
            patch_size // 2 - 1 - padding,
            x0 - padding,
            image.width - x0 - 1 - padding
        ))

        max_perturb_y = max(0, min(
            patch_size // 2 - 1 - padding,
            y0 - padding,
            image.height - y0 - 1 - padding
        ))

        perturb_x = random_symmetric(max_perturb_x)
        perturb_y = random_symmetric(max_perturb_y)

    # Apply perturbation and ensure new coordinates are within valid bounds
    new_x, new_y = x0 + perturb_x, y0 + perturb_y

    # Ensure new keypoint is within the image bounds
    new_x = max(0, min(new_x, image.width - patch_size // 2 - 1 - padding))
    new_y = max(0, min(new_y, image.height - patch_size // 2 - 1 - padding))

    # Crop centered at the new keypoint
    left, upper, right, lower = get_patch_boundary(image, (new_x, new_y), patch_size)
    patch = image.crop((left, upper, right, lower))

    # Convert old keypoint to patch coordinates
    reference = (x0 - left, y0 - upper)

    # Clamp keypoints to patch bounds
    reference_x, reference_y = reference 
    reference_x = max(0, min(reference_x, patch_size - 1))
    reference_y = max(0, min(reference_y, patch_size - 1))
    reference = reference_x, reference_y 
    
    return patch, reference


def prepare_jitter_target_patch(image, keypoint1, keypoint2, patch_size=128, stage='train', valid=True):
    x0, y0 = keypoint1
    x1, y1 = keypoint2

    perturb_x, perturb_y = 0, 0
    padding = config.image.patch_padding 

    if config.image.jitter_target and stage == 'train' and valid:
        max_perturb_x = max(0, min(
            patch_size // 2 - 1 - padding,
            x0 - padding, image.width - x0 - 1 - padding,
            x1 - padding, image.width - x1 - 1 - padding,
        ))

        max_perturb_y = max(0, min(
            patch_size // 2 - 1 - padding,
            y0 - padding, image.height - y0 - 1 - padding,
            y1 - padding, image.height - y1 - 1 - padding,
        ))

        perturb_x = random_symmetric(max_perturb_x)
        perturb_y = random_symmetric(max_perturb_y)

    # Center between keypoints, adjusted by perturbation
    new_x, new_y = (x0 + x1) // 2 + perturb_x, (y0 + y1) // 2 + perturb_y  

    # Ensure new center is within image bounds
    new_x = max(0, min(new_x, image.width - patch_size // 2 - 1 - padding))
    new_y = max(0, min(new_y, image.height - patch_size // 2 - 1 - padding))

    # Crop centered at the new position
    left, upper, right, lower = get_patch_boundary(image, (new_x, new_y), patch_size)
    patch = image.crop((left, upper, right, lower))

    # Convert original keypoints to patch coordinates
    target = (x0 - left, y0 - upper)
    guess = (x1 - left, y1 - upper)

    # Clamp keypoints to patch bounds
    
    target_x, target_y = target 
    target_x = max(0, min(target_x, patch_size - 1))
    target_y = max(0, min(target_y, patch_size - 1))
    target = (target_x, target_y) 
    
    guess_x, guess_y = guess 
    guess_x = max(0, min(guess_x, patch_size - 1))
    guess_y = max(0, min(guess_y, patch_size - 1))
    guess = (guess_x, guess_y) 

    return patch, target, guess


class MatchesDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self,
                 stage,
                 df: pd.DataFrame,
                 patch_normalize=None,
                 image_augmentation_no_kp=None,
                 image_augmentation_kp=None):
        self.stage = stage
        self.df = df

        self.image_augmentation_no_kp = image_augmentation_no_kp
        self.image_augmentation_kp = image_augmentation_kp
        self.patch_normalize = patch_normalize

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _get_image(image_path):
        mode = 'L'
        image = Image.open(image_path).convert(mode)

        return image

    def _prepare_image(self, idx):
        if self.stage in ['val', 'test']:
            random.seed(idx)
        
        row = self.df.iloc[idx, :].values

        [
            DATASET,
            cam,
            kpid,
            pair_name,
            x0, y0,
            x1, y1,
            x_guess, y_guess,
            certainty,
            valid,
        ] = row

        round_digits = 2
        x0, y0, x1, y1, x_guess, y_guess = round(x0, round_digits), round(y0, round_digits), round(x1, round_digits), round(y1, round_digits), round(x_guess, round_digits), round(y_guess, round_digits)
        assert all(not pd.isna(v) for v in [x0, y0, x1, y1, x_guess, y_guess]), f"NaN incoming!"
        
        ref_keypoint = (x0, y0)
        tar_keypoint = (x1, y1)
        guess_keypoint = (x_guess, y_guess)
        valid = bool(valid)

        left_name, right_name = pair_name.split("_")
                
        image_a_path = f"/home/stud/ath/ath_ws/datasets/monado_slam/{DATASET}/mav0/cam{cam}/data/{left_name}.png"
        image_b_path = f"/home/stud/ath/ath_ws/datasets/monado_slam/{DATASET}/mav0/cam{cam}/data/{right_name}.png"

        ref_image = self._get_image(image_a_path)
        tar_image = self._get_image(image_b_path)
               
        if not valid or certainty < config.image.bad_patch_min_confidence:
            certainty = np.float64(0.0)
        else:
            certainty = np.float64(1.0)
                     
        ref_patch, reference = prepare_jitter_reference_patch(
            ref_image, ref_keypoint,
            patch_size=config.image.patch_size,
            stage=self.stage,
        )
        
        tar_patch, target, guess = prepare_jitter_target_patch(
            tar_image, tar_keypoint, guess_keypoint,
            patch_size=config.image.patch_size,
            stage=self.stage,
            valid=valid,
        )
            
        if self.image_augmentation_no_kp:
            patch_np = np.array(ref_patch)
            transformed = self.image_augmentation_no_kp(image=patch_np)
            ref_patch = Image.fromarray(transformed['image'])

            patch_np = np.array(tar_patch)
            transformed = self.image_augmentation_no_kp(image=patch_np)
            tar_patch = Image.fromarray(transformed['image'])

        if self.patch_normalize:
            ref_patch = self.patch_normalize(ref_patch)
            # ref_patch = min_max_normalize(ref_patch, min_val=0.0, max_val=1.0)

            tar_patch = self.patch_normalize(tar_patch)
            # tar_patch = min_max_normalize(tar_patch, min_val=0.0, max_val=1.0)
        
        assert 0 <= reference[0] < config.image.patch_size, f"Reference X-coordinate out of bounds: {reference[0]} (Expected between 0 and {config.image.patch_size})"
        assert 0 <= reference[1] < config.image.patch_size, f"Reference Y-coordinate out of bounds: {reference[1]} (Expected between 0 and {config.image.patch_size})"
        assert 0 <= target[0] < config.image.patch_size, f"Target X-coordinate out of bounds: {target[0]} (Expected between 0 and {config.image.patch_size})"
        assert 0 <= target[1] < config.image.patch_size, f"Target Y-coordinate out of bounds: {target[1]} (Expected between 0 and {config.image.patch_size})"
        assert 0 <= guess[0] < config.image.patch_size, f"Guess X-coordinate out of bounds: {guess[0]} (Expected between 0 and {config.image.patch_size})"
        assert 0 <= guess[1] < config.image.patch_size, f"Guess Y-coordinate out of bounds: {guess[1]} (Expected between 0 and {config.image.patch_size})"
        
        return ref_patch, reference, tar_patch, target, guess, certainty

    def __getitem__(self, idx):
        items = self._prepare_image(idx)
        return items

    @staticmethod
    def _warp_to_pixel_coords(warp):
        desired_patch_size = config.image.patch_size
        w, h = desired_patch_size, desired_patch_size

        warp1 = warp[..., :2]
        warp1 = (
            torch.stack(
                (
                    w * (warp1[..., 0] + 1) / 2,
                    h * (warp1[..., 1] + 1) / 2,
                ),
                axis=-1
            )
        )

        warp2 = warp[..., 2:]
        warp2 = (
            torch.stack(
                (
                    w * (warp2[..., 0] + 1) / 2,
                    h * (warp2[..., 1] + 1) / 2,
                ),
                axis=-1
            )
        )

        return torch.cat((warp1, warp2), dim=-1)


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.experiment.eda_mode else os.cpu_count()
        self.persistent_workers = not config.experiment.eda_mode

        self.image_augmentation_no_kp = A.Compose(
            transforms=[
                A.GaussNoise(p=0.5, std_range=(0.02, 0.04), noise_scale_factor=0.2),
                A.Defocus(p=0.5, radius=2),
            ]
        )

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.dataset: Dict[str, MatchesDataset] = {}

    def setup(self, stage=None):
        if stage == "fit":
            df = pd.read_csv(config.paths.csv.train)
            df = df[df["certainty"] > config.image.patch_min_confidence]

            valid_df = df[df["valid"] == True]
            invalid_df = df[df["valid"] == False]
            invalid_df = invalid_df.sample(frac=0.25, random_state=42)
            df = pd.concat([valid_df, invalid_df], ignore_index=True)
            
            df = df.sample(frac=1).reset_index(drop=True)
            
            excess = len(df) % config.train.batch_size
            if excess:
                df = df.iloc[:-excess]
            
            logger.info(f'Train Length : {len(df)}')

            self.dataset['train'] = MatchesDataset(
                stage="train",
                df=df,
                patch_normalize=self.patch_normalize,
                # image_augmentation_no_kp=self.image_augmentation_no_kp,
            )

            df = pd.read_csv(config.paths.csv.val)
            df = df[df["certainty"] > config.image.patch_min_confidence]
            df = df.sample(frac=1).reset_index(drop=True)
            
            excess = len(df) % config.train.batch_size
            if excess:
                df = df.iloc[:-excess]
                
            logger.info(f'Val Length : {len(df)}')

            self.dataset['val'] = MatchesDataset(
                stage="val",
                df=df,
                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Train Dataset       : {len(self.dataset['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.dataset['val'])} samples")

        if stage == "test":
            df = pd.read_csv(config.paths.csv.test)
            df = df[df["certainty"] > config.image.patch_min_confidence]
            df = df.sample(frac=1).reset_index(drop=True)
            
            excess = len(df) % config.train.batch_size
            if excess:
                df = df.iloc[:-excess]
                
            logger.info(f'Test Length : {len(df)}')

            self.dataset['test'] = MatchesDataset(
                stage="test",
                df=df,
                patch_normalize=self.patch_normalize,
            )

            logger.info(f"Test Dataset  : {len(self.dataset['test'])} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['train'],
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            collate_fn=match_collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset['val'],
            batch_size=config.train.batch_size,
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
