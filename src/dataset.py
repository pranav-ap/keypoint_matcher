from config import config
from utils import logger, min_max_normalize

import os
import gc
from dataclasses import dataclass
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
import scipy

torch.set_float32_matmul_precision('medium')


def print_hdf5_structure(f):
    def print_group(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    try:
        f.visititems(print_group)
    except RuntimeError as e:
        print(f"Skipping corrupted object: {e}")


def crop_image_alb(image: Image.Image, left, upper, right, lower, patch_size=32):
    patch = image.crop((left, upper, right, lower))
    assert patch.size[0] == patch.size[1]
    assert patch.size[0] == patch_size

    return patch


def match_collate_fn(batch):
    ref_patches, references, tar_patches, targets, certainties, cert, estimates = zip(*batch)
    # ref_patches, references, tar_patches, targets, certainties, cert = zip(*batch)

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
        certainties,
        torch.stack(cert).unsqueeze(1),
        estimates,
    )


def weighted_confidence(confidences, x, y, sigma=3):
    """
    Lower sigma = more local focus.
    """
    if confidences.is_cuda:
        confidences = confidences.cpu().numpy()
    else:
        confidences = confidences.numpy()

    H, W = confidences.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Create a Gaussian weight mask centered at (x, y)
    weights = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()  # Normalize to sum to 1

    # Compute weighted confidence
    overall_confidence = np.sum(confidences * weights)

    return overall_confidence


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
        self._file_all_missing = h5py.File(config.paths.matches.all_missing, mode='r')

        self.names = self._get_names()

    def _get_names(self):
        names = []
        ds_types = ['all_missing']

        logger.info(f'Processing {self.stage}')

        for ds_type in ds_types:
            f = self._file_all_missing

            # print_hdf5_structure(f)

            logger.info(f'Processing {ds_type}')

            for video in config.task.videos[self.stage][ds_type]:
                config.task.video = video

                for cam in config.task.cams:
                    config.task.cam = cam

                    logger.info(f'Track & Cam : {video} {cam}')

                    warp_group = f[f'{video}/{cam}/matcher/warp']
                    cert_group = f[f'{video}/{cam}/matcher/certainty']
                    saves_group = f[f'{video}/{cam}/matcher/saves']

                    for pair_name in warp_group.keys() & cert_group.keys() & saves_group.keys():
                        warp_dataset = warp_group[pair_name]
                        cert_dataset = cert_group[pair_name]
                        saves_dataset = saves_group[pair_name]

                        if not isinstance(warp_dataset, h5py.Dataset) or not isinstance(cert_dataset, h5py.Dataset) or not isinstance(saves_dataset, h5py.Dataset):
                            continue

                        image_name_a, image_name_b, kpid = pair_name.split('_')

                        warp = warp_dataset[()]

                        try:
                            cert = cert_dataset[()]
                            cert = torch.from_numpy(cert)
                        except Exception as e:
                            logger.debug('oops')
                            continue

                        saves = saves_dataset[()]

                        assert warp.shape == (config.image.patch_size, config.image.patch_size, 4), f'{warp.shape=}'
                        assert cert.shape == (config.image.patch_size, config.image.patch_size), f'{cert.shape=}'
                        assert len(saves) == 12, f'{len(saves)=}' # 10 12

                        reference = estimate = [0, 0]

                        [
                            reference[0], reference[1],
                            ref_left, ref_upper, ref_right, ref_lower,

                            estimate[0], estimate[1],
                            tar_left, tar_upper, tar_right, tar_lower,
                        ] = saves

                        desired_patch_size = 82

                        if not (0 <= reference[0] < desired_patch_size and 0 <= reference[1] < desired_patch_size):
                            logger.debug(f'Bad ref crop kp {reference=}')
                            continue

                        if not (0 <= estimate[0] < desired_patch_size and 0 <= estimate[1] < desired_patch_size):
                            logger.debug(f'Bad tar crop kp {estimate=}')
                            continue

                        x, y = reference
                        y0, x0 = int(y), int(x)

                        # certainty = cert[y0, x0]
                        certainty = weighted_confidence(cert, x0, y0, sigma=2)

                        # gaussian_cert = scipy.ndimage.gaussian_filter(cert, sigma=2)
                        # certainty = gaussian_cert[y0, x0]

                        certainty = round(certainty.item(), 2)

                        if certainty > 0.01:
                            names.append((video, cam, image_name_a, image_name_b, kpid, saves, certainty))

                        # if len(names) > 600:
                        #     break

                logger.info(f'{len(names)=}')

        excess = len(names) % config.train.train_batch_size
        if excess:
            names = names[:-excess]

        return names

    def __len__(self):
        if not hasattr(self, '_file_all_missing'):
            self._setup_file()

        assert hasattr(self, '_file_all_missing')

        return len(self.names)

    @staticmethod
    def crop_from_center(image, crop_width, crop_height):
        img_width, img_height = image.size

        center_x, center_y = img_width // 2, img_height // 2

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2
        right = center_x + crop_width // 2
        bottom = center_y + crop_height // 2

        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

    def _get_image(self, image_name):
        image_path = os.path.join(config.paths.images, f'{image_name}.png')
        mode = 'L' # 'RGB'
        image = Image.open(image_path).convert(mode)

        # w, h = config.image.crop_image_shape
        # image = self.crop_from_center(image, w, h)

        return image

    @staticmethod
    def get_patch_boundary(image: Image.Image, center_point, patch_size):
        image_width, image_height = image.size
        x, y = center_point
        half_patch_size = patch_size // 2

        left = max(0, min(x - half_patch_size, image_width - patch_size))
        upper = max(0, min(y - half_patch_size, image_height - patch_size))

        right, lower = left + patch_size, upper + patch_size

        assert right > left
        assert right - left == patch_size
        assert lower > upper
        assert lower - upper == patch_size

        return left, upper, right, lower

    @staticmethod
    def _get_patch_boundary2(image: Image.Image, center_point, patch_size):
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
        try:
            transform = A.Compose(
                transforms=[
                    A.Crop(
                        x_min=round(left), y_min=round(upper),
                        x_max=round(right), y_max=round(lower),
                    ),
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        except Exception as e:
            logger.debug(f'{keypoint, left, upper, right, lower=}')
            print(e)

        transformed = transform(image=np.array(image), keypoints=[keypoint])

        patch = Image.fromarray(transformed['image'])
        assert patch.size[0] == patch.size[1], f'patch must be a square : {patch.size=}'
        assert patch.size[0] == right - left
        assert patch.size[1] == lower - upper

        keypoints = transformed['keypoints']
        assert len(keypoints) == 1, "Expected a single transformed keypoint"
        keypoint = keypoints[0][0], keypoints[0][1]

        return patch, keypoint

    def _warp_to_pixel_coords(self, warp):
        desired_patch_size = config.image.patch_size
        w, h = desired_patch_size, desired_patch_size # config.image.crop_image_shape

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

    def _prepare_patch(self, image_name, left, upper, right, lower):
        image = self._get_image(image_name)
        desired_patch_size = config.image.train_patch_size

        patch = crop_image_alb(
           image, left, upper, right, lower, patch_size=desired_patch_size
        )

        if self.stage == 'train' and self.image_augmentation_no_kp:
            patch_np = np.array(patch)
            transformed = self.image_augmentation_no_kp(
                image=patch_np,
            )

            patch = Image.fromarray(transformed['image'])

        if self.patch_normalize:
            patch = self.patch_normalize(patch)
            patch = min_max_normalize(patch, min_val=0.0, max_val=1.0)

        return patch

    def _prepare_image(self, idx):
        name = self.names[idx]
        video, cam, image_name_a, image_name_b, kpid, saves, certainty = name

        config.task.video = video
        config.task.cam = cam

        f = self._file_all_missing

        warp_group = f[f'{video}/{cam}/matcher/warp']
        cert_group = f[f'{video}/{cam}/matcher/certainty']

        pair_name = f'{image_name_a}_{image_name_b}_{kpid}'
        warp = warp_group[pair_name][()].astype(np.float32)
        cert = cert_group[pair_name][()].astype(np.float32)

        warp = torch.from_numpy(warp)
        cert = torch.from_numpy(cert)
        pixel_coords = self._warp_to_pixel_coords(warp)

        reference = estimate = [0, 0]

        [
            reference[0], reference[1],
            ref_left, ref_upper, ref_right, ref_lower,

            estimate[0], estimate[1],
            tar_left, tar_upper, tar_right, tar_lower,
        ] = saves

        ref_patch = self._prepare_patch(image_name_a, ref_left, ref_upper, ref_right, ref_lower)
        tar_patch = self._prepare_patch(image_name_b, tar_left, tar_upper, tar_right, tar_lower)

        x0, y0 = reference
        y0, x0 = int(y0.item()), int(x0.item())

        x0, y0, x1, y1 = pixel_coords[y0, x0]
        x1, y1 = x1.item(), y1.item()

        reference = (x0, y0)
        target = (x1, y1)

        return ref_patch, reference, tar_patch, target, certainty, cert, estimate

    def __getitem__(self, idx):
        if not hasattr(self, '_file_all_missing'):
            self._setup_file()

        assert hasattr(self, '_file_all_missing')

        ref_patch, reference, tar_patch, target, certainty, cert, estimate = self._prepare_image(idx)

        return ref_patch, reference, tar_patch, target, certainty, cert, estimate


class MatchesDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.task.eda_mode else os.cpu_count()
        self.persistent_workers = not config.task.eda_mode

        self.image_augmentation_no_kp = A.Compose(
            transforms=[
                # A.GaussNoise(p=0.7, std_range=(0.04, 0.07), noise_scale_factor=0.5),
                A.Defocus(p=0.3, radius=1),
            ]
        )

        # self.patch_normalize = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ])

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=[0.5], std=[0.5]),
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
