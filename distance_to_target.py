from config import config
from utils import logger
import os
import gc 
import pandas as pd
import glob
import os
import numpy as np
import h5py
import torch 


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


def main():
    f = h5py.File(config.paths.matches.all_missing, mode='r')
    stage = 'train'
    ds_type = 'all_missing'

    errors = []

    for video in config.task.videos[stage][ds_type]:
        config.task.video = video
        
        track_errors = []

        for cam in config.task.cams:
            config.task.cam = cam

            logger.info(f'Track & Cam : {video} {cam}')

            warp_group = f[f'{video}/{cam}/matcher/warp']
            saves_group = f[f'{video}/{cam}/matcher/saves']

            for pair_name in warp_group.keys() & saves_group.keys():
                warp_dataset = warp_group[pair_name]
                saves_dataset = saves_group[pair_name]

                if not isinstance(warp_dataset, h5py.Dataset) or not isinstance(saves_dataset, h5py.Dataset):
                    continue

                warp = warp_dataset[()].astype(np.float32)
                saves = saves_dataset[()].astype(np.float32)
                assert len(saves) == 12, f'{len(saves)=}'

                warp = torch.from_numpy(warp)
                pixel_coords = _warp_to_pixel_coords(warp)

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
                
                x0, y0, x1, y1 = pixel_coords[y0, x0]
                x1, y1 = x1.item(), y1.item()

                reference = (x0, y0)
                target = (x1, y1)
                
                # Calculate the Euclidean distance error
                error = np.sqrt((x1 - estimate[0]) ** 2 + (y1 - estimate[1]) ** 2)
                
                track_errors.append(error)
                errors.append(error)
        
        mean_error = np.mean(track_errors) if track_errors else float('nan')
        std_error = np.std(track_errors) if track_errors else float('nan')
        min_error = np.min(track_errors)
        max_error = np.max(track_errors)
            
        logger.info(
            f"Mean Distance Error {video}: {mean_error:.4f}, "
            f"Std Dev: {std_error:.4f}, Min: {min_error:.4f}, Max: {max_error:.4f} "
            f"(Samples: {len(track_errors)})"
        )

    f.close()

    mean_error = np.mean(errors) if errors else float('nan')
    std_error = np.std(errors) if errors else float('nan')
    min_error = np.min(errors)
    max_error = np.max(errors)
        
    logger.info(
        f"Overall Mean Distance Error: {mean_error:.4f}, "
        f"Std Dev: {std_error:.4f}, Min: {min_error:.4f}, Max: {max_error:.4f} "
        f"(Total Samples: {len(errors)})"
    )
    

if __name__ == '__main__':
    main()
