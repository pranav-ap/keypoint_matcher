import pandas as pd
import glob
import os
import numpy as np



"""

=> track_debug_3_lifetimes

len(combined_df)=405,147

Estimate is beyond 90 pixels: 0.60%
Estimate is beyond 60 pixels: 3.3%
Estimate is beyond 30 pixels: 17.78%
Estimate is beyond 25 pixels: 23.63%
Estimate is beyond 5 pixels: 74.51%
Estimate is beyond 2 pixels: 87.56%
Estimate is beyond 1.5 pixels: 91.84%
Estimate is beyond 1.25 pixels: 91.84%
Estimate is beyond 1 pixel: 91.84%




=> track_debug

len(combined_df)=2,464,266

Estimate is beyond 90 pixels: 1.37%
Estimate is beyond 60 pixels: 5.56%
Estimate is beyond 30 pixels: 19.60%
Estimate is beyond 25 pixels: 24.78%
Estimate is beyond 5 pixels: 74.75%
Estimate is beyond 2 pixels: 88.62%
Estimate is beyond 1.5 pixels: 92.96%
Estimate is beyond 1.25 pixels: 92.96%
Estimate is beyond 1 pixel: 92.96%

"""



tracks = [
    'MOO01_hand_puncher_1', # re for track_debug
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


def compute_coords_accuracy_percentage(estimates, target, pixels=1):
    difference = np.abs(estimates - target)
    # Check which differences exceed N pixels
    exceeds = np.any(difference > pixels, axis=1)
    percentage = (np.sum(exceeds) / estimates.shape[0]) * 100
    return percentage


def compute_error_stats(estimates, targets):
    # Calculate Euclidean distance between predicted and target coordinates
    error = np.sqrt(np.sum((estimates - targets) ** 2, axis=1))
    mean_error = np.mean(error)
    std_error = np.std(error)
    return mean_error, std_error


def main():
    filter_type = 'track_debug_3_lifetimes' # track_debug_3_lifetimes track_debug
    all_data = []    

    for track in tracks:
        print(f'Loading {track=}')

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

    estimates = combined_df[["x_guess", "y_guess"]].to_numpy(dtype=np.float32)
    targets = combined_df[["x", "y"]].to_numpy(dtype=np.float32)

    coords_percent_90_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=90)
    coords_percent_60_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=60)
    coords_percent_30_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=30)
    coords_percent_25_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=25)
    coords_percent_5_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=5)
    coords_percent_2_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=2)
    coords_percent_15_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=1.5)
    coords_percent_125_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=1.25)
    coords_percent_1_pixel = compute_coords_accuracy_percentage(estimates, targets, pixels=1)

    mean_error, std_error = compute_error_stats(estimates, targets)

    print(f"Accuracy within 90 pixels: {coords_percent_90_pixel:.2f}%")
    print(f"Accuracy within 60 pixels: {coords_percent_60_pixel:.2f}%")
    print(f"Accuracy within 30 pixels: {coords_percent_30_pixel:.2f}%")
    print(f"Accuracy within 25 pixels: {coords_percent_25_pixel:.2f}%")
    print(f"Accuracy within 5 pixels: {coords_percent_5_pixel:.2f}%")
    print(f"Accuracy within 2 pixels: {coords_percent_2_pixel:.2f}%")
    print(f"Accuracy within 1.5 pixels: {coords_percent_15_pixel:.2f}%")
    print(f"Accuracy within 1.25 pixels: {coords_percent_125_pixel:.2f}%")
    print(f"Accuracy within 1 pixel: {coords_percent_1_pixel:.2f}%")
    
    print(f"Mean error (Euclidean distance): {mean_error:.2f} pixels")
    print(f"Standard deviation of error: {std_error:.2f} pixels")

if __name__ == '__main__':
    main()





"""
=> track_debug_3_lifetimes

len(combined_df)=405147

Accuracy within 2 pixels: 87.56%
Accuracy within 1.5 pixels: 91.84%
Accuracy within 1.25 pixels: 91.84%
Accuracy within 1 pixel: 91.84%

Mean error (Euclidean distance): 19.27 pixels
Standard deviation of error: 18.48 pixels

=> track_debug

len(combined_df)=2464266

Accuracy within 2 pixels: 88.62%
Accuracy within 1.5 pixels: 92.96%
Accuracy within 1.25 pixels: 92.96%
Accuracy within 1 pixel: 92.96%

Mean error (Euclidean distance): 20.81 pixels
Standard deviation of error: 21.75 pixels

"""


