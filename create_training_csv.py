import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import h5py
import torch
from utils import logger


class Reader:
    def __init__(self, cam: int, DATASET: str):
        DATASET_PATH = f"D:/thesis_code/datasets/output/output_all/basalt/monado_slam/{DATASET}"
        filename = "data.hdf5"  # do not change
        filepath = f"{DATASET_PATH}/{filename}"
        self._file = h5py.File(filepath, "r")

        self.cam = f"cam{cam}"
        assert self.cam in ["cam0", "cam1"]

        # [width, height]
        w = 640
        h = 480
        r = 14
        cw = w // r * r
        ch = h // r * r
        # print(f"{cw=}, {ch=}")
        hpad = (w - cw) // 2
        vpad = (h - ch) // 2
        assert cw + hpad * 2 == w and ch + vpad * 2 == h

        self.original_image_shape = [w, h]
        self.crop_image_shape = [cw, ch]
        self.pad = [hpad, vpad]

        self._init_groups_read_mode()

    def _init_groups_read_mode(self):
        self._matcher = self._file[f"{self.cam}/matcher"]
        self.matcher_warp = self._matcher["warp"]
        self.matcher_certainty = self._matcher["certainty"]

    def close(self):
        self._file.close()

    @staticmethod
    def _warp_to_pixel_coords(warp):
        """
        This function is from a RoMa utils file
        """
        h1, w1 = 476, 630
        h2, w2 = 476, 630

        warp1 = warp[..., :2]
        warp1 = torch.stack(
            (
                w1 * (warp1[..., 0] + 1) / 2,
                h1 * (warp1[..., 1] + 1) / 2,
            ),
            axis=-1,
        )

        warp2 = warp[..., 2:]
        warp2 = torch.stack(
            (
                w2 * (warp2[..., 0] + 1) / 2,
                h2 * (warp2[..., 1] + 1) / 2,
            ),
            axis=-1,
        )

        return torch.cat((warp1, warp2), dim=-1)

    def load_warp(self, pair_name):
        warp = self.matcher_warp[pair_name][()]
        warp = torch.from_numpy(warp)

        pixel_coords = self._warp_to_pixel_coords(warp)
        certainty = self.matcher_certainty[pair_name][()]

        return pixel_coords, certainty

    def get_target_keypoint(self, flow, certainties, x0, y0):
        hpad, vpad = self.pad
        cw, ch = self.crop_image_shape

        if x0 < 0 or x0 > cw or y0 < 0 or y0 > ch:
            return 0, 0, 0

        x0 -= hpad
        y0 -= vpad

        # we int() instead of round() since roma estimates for pixel center and not pixel border
        y0 = int(y0)
        x0 = int(x0)

        _, _, x1, y1 = flow[y0, x0]
        x1, y1 = x1.item(), y1.item()

        certainty = certainties[y0, x0]

        x1 += hpad
        y1 += vpad

        return x1, y1, certainty

    def print_hdf5_structure(self):
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        self._file.visititems(print_group)


def process_cam(cam: int, training_df: pd.DataFrame, DATASET: str):
    reader = Reader(cam, DATASET)

    for pair_name in tqdm(reader.matcher_warp.keys(), desc=f"Extracting from frames", leave=False):
        warp, cert = reader.load_warp(pair_name)
        assert warp is not None, f"Failed to load warp for {pair_name=}"

        left_name = pair_name.split("_")[0]

        file_path = f"D:/thesis_code/track_debug/{DATASET}/cam{cam}/{left_name}_incoming_missed_kps.csv"
        assert os.path.exists(file_path), f"File {file_path} does not exist!"

        df = pd.read_csv(
            file_path,
            header=0,
            names=("kpid", "x", "y", "x_guess", "y_guess")
        )

        for i in range(len(df)):
            kpid, x0, y0, x_guess, y_guess = df.iloc[i, :5]
            x1, y1, certainty = reader.get_target_keypoint(warp, cert, x0, y0)

            training_df.loc[len(training_df)] = [
                DATASET,
                cam,
                kpid,
                pair_name,
                x0, y0,
                x1, y1,
                x_guess, y_guess,
                certainty
            ]

    reader.close()


def main():
    DATASETS = [
        "MOO01_hand_puncher_1",
        "MOO02_hand_puncher_2",
        "MOO03_hand_shooter_easy",
        "MOO04_hand_shooter_hard",
        "MOO05_inspect_easy",
        "MOO06_inspect_hard",
        "MOO07_mapping_easy",
        "MOO08_mapping_hard",
        "MOO09_short_1_updown",
        "MOO10_short_2_panorama",
        "MOO11_short_3_backandforth",
    ]

    training_df = pd.DataFrame(columns=[
        "dataset",
        "cam",
        "kpid",
        "pair_name",
        "x0", "y0",
        "x1", "y1",
        "x_guess", "y_guess",
        "certainty",
    ])

    for DATASET in DATASETS:
        print(f"DATASET: {DATASET}")

        for cam in [0, 1]:
            print(f"=> cam{cam}")
            process_cam(cam, training_df, DATASET)

    training_df.to_csv(f"data/training.csv", index=False)


if __name__ == "__main__":
    main()
