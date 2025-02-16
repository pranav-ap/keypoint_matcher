import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import cv2
import albumentations as A
import h5py
import lightning as L
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config import config
from utils import logger, min_max_normalize

torch.set_float32_matmul_precision('medium')



def crop_image_alb(image: Image.Image, keypoint, left, upper, right, lower, patch_size=32):
    patch = image.crop((left, upper, right, lower))
    assert patch.size[0] == patch.size[1]
    assert patch.size[0] == patch_size

    new_keypoint = keypoint[0] - left, keypoint[1] - upper

    return patch, new_keypoint
    


def match_collate_fn(batch):
    ref_patches, ref_keypoints, tar_patches, tar_keypoints, certainties = zip(*batch)

    # Convert keypoints from list of tuples to tensor
    ref_keypoints = torch.tensor(ref_keypoints, dtype=torch.float32)
    tar_keypoints = torch.tensor(tar_keypoints, dtype=torch.float32)
    certainties = torch.tensor(certainties, dtype=torch.float32)

    return (
        torch.cat(ref_patches, dim=0).unsqueeze(1), 
        ref_keypoints, 
        torch.cat(tar_patches, dim=0).unsqueeze(1), 
        tar_keypoints,
        certainties
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
        self._file_all_missing = h5py.File(config.paths.matches.all_missing, mode='r')

        self.names = self._get_names()

    def _get_names(self):
        names = []
        ds_types = ['all_missing']

        logger.info(f'Processing {self.stage}')

        for ds_type in ds_types:
            f = self._file_all_missing

            logger.info(f'Processing {ds_type}')

            for video in config.task.videos[self.stage][ds_type]:
                config.task.video = video

                for cam in config.task.cams:
                    config.task.cam = cam

                    logger.info(f'Track & Cam : {video} {cam}')

                    warp_group = f[f'{video}/{cam}/matcher/warp']
                    cert_group = f[f'{video}/{cam}/matcher/certainty']

                    store = {
                        'missed_kps_csv_path': None,
                        'df': None
                    }

                    # logger.info(f'{len(warp_group.keys())=}')
                    # logger.info(f'{warp_group.keys()=}')

                    # logger.info(f'{len(cert_group.keys())=}')
                    # logger.info(f'{cert_group.keys()=}')

                    # logger.info(f'{len((warp_group.keys() & cert_group.keys()))=}')
                    # logger.info(f'{(warp_group.keys() & cert_group.keys())=}')
                    
                    for pair_name in warp_group.keys() & cert_group.keys():
                        warp_dataset = warp_group[pair_name]
                        cert_dataset = cert_group[pair_name]

                        if not isinstance(warp_dataset, h5py.Dataset) or not isinstance(cert_dataset, h5py.Dataset):
                            continue
                        
                        image_name_a, image_name_b, kpid = pair_name.split('_')

                        warp = warp_dataset[()]
                        cert = cert_dataset[()]

                        assert warp.shape == (config.image.patch_size, config.image.patch_size, 4), f'{warp.shape=}'
                        assert cert.shape == (config.image.patch_size, config.image.patch_size), f'{cert.shape=}'

                        missed_kps_csv_path = f"/home/stud/ath/ath_ws/datasets/track_debug/{video}/{cam}/{image_name_a}_incoming_missed_kps.csv"

                        if missed_kps_csv_path != store['missed_kps_csv_path']:
                            store['missed_kps_csv_path'] = missed_kps_csv_path

                            store['df'] = pd.read_csv(
                                missed_kps_csv_path,
                                header=0, names=("kpid", "x", "y", "x_guess", "y_guess")
                            )

                        df = store['df']
                        row = df[df['kpid'] == int(kpid)]
                        row = row.iloc[0]
                        x, y, x_guess, y_guess = row["x"], row["y"], row["x_guess"], row["y_guess"]

                        names.append((video, cam, image_name_a, image_name_b, kpid, x, y, x_guess, y_guess))

                        # im = self._get_image(image_name_a)

                        # image_width, image_height = im.size
                        # desired_patch_size = config.image.patch_size
                        # assert desired_patch_size < image_width
                        # assert desired_patch_size < image_height

                        # keypoint = x, y

                        # left, upper, right, lower = self._get_patch_boundary(
                        #     im,
                        #     keypoint,
                        #     patch_size=desired_patch_size
                        # )

                        # patch, keypoint = crop_image_alb(
                        #     im, [x, y], left, upper, right, lower, patch_size=desired_patch_size
                        # )

                        # x0, y0 = keypoint
                        
                        # y0 = int(y0)
                        # x0 = int(x0)

                        # certainty = cert[y0, x0]
                        # certainty_threshold = 0.5

                        # if certainty > certainty_threshold or np.random.rand() < certainty:
                        #     names.append((video, cam, image_name_a, image_name_b, kpid, x, y, x_guess, y_guess))
                        
        # print(names)

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

    def _prepare_patch(self, image_name, x, y, image_idx):
        image = self._get_image(image_name)

        image_width, image_height = image.size
        padded_patch_size = config.image.patch_size
        assert padded_patch_size < image_width
        assert padded_patch_size < image_height

        keypoint = x, y

        left, upper, right, lower = self._get_patch_boundary(
            image,
            keypoint,
            patch_size=padded_patch_size
        )

        patch, keypoint = crop_image_alb(
           image, [x, y], left, upper, right, lower, patch_size=padded_patch_size
        )
        
        # np.random.seed(image_idx)

        center_point = x, y = keypoint

        desired_patch_size = config.image.train_patch_size

        perturb_size = config.image.patch_border # (desired_patch_size - config.image.patch_border) // 2

        perturb_x = np.random.randint(1, perturb_size)
        perturb_y = np.random.randint(1, perturb_size)

        perturb_x = perturb_x * np.random.choice([1, -1])
        perturb_y = perturb_y * np.random.choice([1, -1])

        if 0 <= x + perturb_x < desired_patch_size and 0 <= y + perturb_y < desired_patch_size:
            center_point = x + perturb_x, y + perturb_y

        left, upper, right, lower = self._get_patch_boundary(
            patch,
            center_point,  # center of new smaller patch
            desired_patch_size
        )

        patch, keypoint = crop_image_alb(
           patch, 
           keypoint, # the keypoint that gets transformed into smaller patch
           left, upper, right, lower, 
           patch_size=desired_patch_size
        )       
        
        if self.patch_normalize:
            patch = self.patch_normalize(patch)
            # patch = min_max_normalize(patch, min_val=0.0, max_val=1.0) 

        return patch, keypoint

    def _prepare_image(self, idx):
        name = self.names[idx]
        video, cam, image_name_a, image_name_b, kpid, x, y, x_guess, y_guess = name

        # logger.debug(f'{video, cam, image_name_a, image_name_b, kpid, x, y, x_guess, y_guess=}')

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

        ref_patch, ref_keypoint = self._prepare_patch(image_name_a, x, y, idx)
        tar_patch, tar_keypoint = self._prepare_patch(image_name_b, x_guess, y_guess, idx)

        x, y = ref_keypoint

        y0 = int(y)
        x0 = int(x)

        certainty = cert[y0, x0]

        x0, y0, x1, y1 = pixel_coords[y0, x0]
        x1, y1 = x1.item(), y1.item()

        ref_keypoint = (x0, y0)
        tar_keypoint = (x1, y1)

        return ref_patch, ref_keypoint, tar_patch, tar_keypoint, certainty

    def __getitem__(self, idx):
        if not hasattr(self, '_file_all_missing'):
            self._setup_file()

        assert hasattr(self, '_file_all_missing')

        ref_patch, ref_keypoint, tar_patch, tar_keypoint, certainty = self._prepare_image(idx)
        
        return ref_patch, ref_keypoint, tar_patch, tar_keypoint, certainty


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

        # self.patch_normalize = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ])

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
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
