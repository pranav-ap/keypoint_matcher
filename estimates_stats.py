import pandas as pd
import glob
import os
import numpy as np


tracks = [
    'MOO01_hand_puncher_1', # remove for track_debug
    'MOO02_hand_puncher_2', 
    'MOO03_hand_shooter_easy', 
    'MOO04_hand_shooter_hard',
    'MOO05_inspect_easy',
    'MOO06_inspect_hard',
    'MOO07_mapping_easy',
    'MOO08_mapping_hard',
    'MOO09_short_1_updown',
    'MOO10_short_2_panorama',
    'MOO11_short_3_backandforth',
]


def compute_coords_accuracy_percentage(coords_pred, target, pixels=1):
    difference = np.abs(coords_pred - target)
    exceeds = np.any(difference > pixels, axis=1)
    percentage = (np.sum(exceeds) / coords_pred.shape[0]) * 100
    return percentage


def compute_average_error(coords_pred, targets):
    # Calculate Euclidean distance between predicted and target coordinates
    error = np.sqrt(np.sum((coords_pred - targets) ** 2, axis=1))
    average_error = np.mean(error)
    return average_error



def main():
    filter_type = 'track_debug' # track_debug_3_lifetimes track_debug
    all_data = []    

    for track in tracks:
        for cam in ['cam0', 'cam1']:
            folder_path = f"/home/stud/ath/ath_ws/datasets/{filter_type}/{track}/{cam}/"
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

            for file in csv_files:
                df = pd.read_csv(
                    file,
                    header=0, 
                    names=("kpid", "x", "y", "x_guess", "y_guess")
                )

                all_data.append(df)

    if not all_data:
        print("No data found.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True).dropna()

    print(f'len(combined_df)={len(combined_df)}')

    coords_pred = combined_df[["x_guess", "y_guess"]].to_numpy(dtype=np.float32)
    targets = combined_df[["x", "y"]].to_numpy(dtype=np.float32)

    coords_percent_2_pixel = compute_coords_accuracy_percentage(coords_pred, targets, pixels=2)
    coords_percent_15_pixel = compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
    coords_percent_125_pixel = compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
    coords_percent_1_pixel = compute_coords_accuracy_percentage(coords_pred, targets, pixels=1)

    avg_error = compute_average_error(coords_pred, targets)

    print(f"Accuracy within 2 pixels: {coords_percent_2_pixel:.2f}%")
    print(f"Accuracy within 1.5 pixels: {coords_percent_15_pixel:.2f}%")
    print(f"Accuracy within 1.25 pixels: {coords_percent_125_pixel:.2f}%")
    print(f"Accuracy within 1 pixel: {coords_percent_1_pixel:.2f}%")
    print(f"Average error (Euclidean distance) : {avg_error:.2f} pixels")


if __name__ == '__main__':
    main()


"""
=> track_debug_3_lifetimes

len(combined_df)=405147

Accuracy within 2 pixels: 87.56%
Accuracy within 1.5 pixels: 91.84%
Accuracy within 1.25 pixels: 91.84%
Accuracy within 1 pixel: 91.84%
Average error: 19.27 pixels

=> track_debug

len(combined_df)=2464266

Accuracy within 2 pixels: 88.62%
Accuracy within 1.5 pixels: 92.96%
Accuracy within 1.25 pixels: 92.96%
Accuracy within 1 pixel: 92.96%
Average error (Euclidean distance) : 20.81 pixels

"""