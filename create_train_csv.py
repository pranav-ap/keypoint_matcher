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


tracks = [
    # 'MOO01_hand_puncher_1', 
    # 'MOO02_hand_puncher_2', 
    # 'MOO03_hand_shooter_easy', 
    # 'MOO04_hand_shooter_hard',
    # 'MOO05_inspect_easy',
    # 'MOO06_inspect_hard',
    # 'MOO07_mapping_easy',
    # 'MOO08_mapping_hard',
    # 'MOO09_short_1_updown',
    # 'MOO10_short_2_panorama',
    'MOO11_short_3_backandforth',
]



def print_hdf5_structure(f):
    def print_group(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    try:
        f.visititems(print_group)
    except RuntimeError as e:
        print(f"Skipping corrupted object: {e}")
        
        
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



def process(video):
    for cam in config.task.cams:
        config.task.cam = cam

        logger.info(f'Track & Cam : {video} {cam}')

        warp_group = f[f'{cam}/matcher/warp']
        cert_group = f[f'{cam}/matcher/certainty']
        
        for pair_name in warp_group.keys() & cert_group.keys():
            warp_dataset = warp_group[pair_name]
            cert_dataset = cert_group[pair_name]

            if not isinstance(warp_dataset, h5py.Dataset) or not isinstance(cert_dataset, h5py.Dataset):
                continue

            try:
                warp = warp_dataset[()].astype(np.float32)
                warp = torch.from_numpy(warp)

                cert = cert_dataset[()].astype(np.float32)
                cert = torch.from_numpy(cert)

            except Exception as e:
                logger.debug('oops')
                continue

            pixel_coords = _warp_to_pixel_coords(warp)
 


def main():   
    for track in tracks:    
        config.task.video = track
        
        filepath = f'/home/stud/ath/ath_ws/keypoint_dataset_pipeline/output/output_all/basalt/monado_slam/{track}/data.hdf5'
        f = h5py.File(filepath, mode='r')
        
        print_hdf5_structure(f)
        process(track)
        
        f.close()

    return
    
    stages = ['train', 'val']
    

    f.close()
    

if __name__ == '__main__':
    main()
