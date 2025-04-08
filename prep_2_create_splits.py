import os
import shutil
import pandas as pd
import random


def filter_high_error_kpids(training_df, high_error_kpids_df):
    """Filter training data by removing rows corresponding to high error keypoints."""
    high_error_kpids = high_error_kpids_df["kpid"].values
    clean_df = training_df[~training_df["kpid"].isin(high_error_kpids)]
    return clean_df

def mark_high_error_kpids(training_df, high_error_kpids_df):
    """Mark rows with high error kpids instead of removing them."""
    high_error_kpids = set(high_error_kpids_df["kpid"].values)
    training_df["valid"] = ~training_df["kpid"].isin(high_error_kpids)  # True if not high error, False otherwise
    return training_df


def generate_random_keypoint(image_width, image_height, patch_margin):
    x = random.uniform(patch_margin, image_width - patch_margin)
    y = random.uniform(patch_margin, image_height - patch_margin)
    return x, y


def augment_with_random_patches(df, image_width=640, image_height=480):
    new_rows = []
    for _, row in df.iterrows():
        x1, y1 = generate_random_keypoint(image_width, image_height, patch_margin=52)
        x_guess, y_guess = x1, y1
        # x_guess = x1 + random.randint(-2, 2)
        # y_guess = y1 + random.randint(-2, 2)

        new_row = row.copy()
        new_row["x1"] = x1
        new_row["y1"] = y1
        new_row["x_guess"] = x_guess
        new_row["y_guess"] = y_guess
        new_row["certainty"] = 0.0
        new_row["valid"] = False

        new_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def augment_with_random_patches_inline(df, image_width=640, image_height=480):
    df = df.copy()

    x1_list, y1_list, xg_list, yg_list = [], [], [], []

    for _ in df.itertuples():
        x1, y1 = generate_random_keypoint(image_width, image_height, patch_margin=52)
        x1_list.append(x1)
        y1_list.append(y1)
        xg_list.append(x1)  # or add slight offset if you want
        yg_list.append(y1)

    df["x1_fake"] = x1_list
    df["y1_fake"] = y1_list
    df["x_guess_fake"] = xg_list
    df["y_guess_fake"] = yg_list
    df["certainty_fake"] = 0.0

    return df


def create_datasets(df, splits):
    train_df = df[df["dataset"].isin(splits['train'])]
    val_df = df[df["dataset"].isin(splits['val'])]
    test_df = df[df["dataset"].isin(splits['test'])]

    return train_df, val_df, test_df


def print_dataset_stats(clean_train_df, clean_val_df, clean_test_df):
    """Print the statistics of rows per dataset and overall with percentages."""
    train_size = len(clean_train_df)
    val_size = len(clean_val_df)
    test_size = len(clean_test_df)
    total_size = train_size + val_size + test_size

    print("=> Dataset Stats:")
    print(f"Train dataset size: {train_size} ({(train_size / total_size) * 100:.2f}%)")
    print(f"Validation dataset size: {val_size} ({(val_size / total_size) * 100:.2f}%)")
    print(f"Test dataset size: {test_size} ({(test_size / total_size) * 100:.2f}%)")
    print(f"Total cleaned dataset size: {total_size} (100%)")


def main(threshold=30, patch_size=32):
    splits = {
        'train': [
            "MOO01_hand_puncher_1",
            "MOO02_hand_puncher_2",
            "MOO03_hand_shooter_easy",
            "MOO04_hand_shooter_hard",
            "MOO05_inspect_easy",
        ],
        'val': [
            "MOO06_inspect_hard",
            "MOO07_mapping_easy",
            "MOO08_mapping_hard",
            "MOO09_short_1_updown",
            "MOO10_short_2_panorama",
        ],
        'test': [
            "MOO11_short_3_backandforth",
        ]
    }

    training_df = pd.read_csv("/home/stud/ath/ath_ws/datasets/match_april/training.csv")
    high_error_kpids_df = pd.read_csv("/home/stud/ath/ath_ws/datasets/match_april/high_error_kpids.csv")

    print(f'{len(training_df)=}')

    # clean_df = mark_high_error_kpids(training_df, high_error_kpids_df)
    clean_df = filter_high_error_kpids(training_df, high_error_kpids_df)

    print(f'{len(clean_df)=}')

    image_width = 640    
    image_height = 480

    # Keep only rows where (x0, y0) have at least patch_size margin to all sides
    # centered_df = clean_df
    
    centered_df = clean_df[
        (clean_df["x0"] >= patch_size // 2) &
        (clean_df["y0"] >= patch_size // 2) &
        (clean_df["x0"] <= image_width - patch_size // 2 - 1) &
        (clean_df["y0"] <= image_height - patch_size // 2 - 1)
    ]

    print(f'{len(centered_df)=}')

    clean_train_df, clean_val_df, clean_test_df = create_datasets(centered_df, splits)

    clean_train_df.loc[:, "valid"] = True
    clean_val_df.loc[:, "valid"] = True
    clean_test_df.loc[:, "valid"] = True

    # clean_train_df = augment_with_random_patches(clean_train_df, image_width, image_height)
    clean_train_df = augment_with_random_patches_inline(clean_train_df, image_width, image_height)
    
    valid_count = clean_train_df["valid"].sum()  # Count of True values (valid rows)
    invalid_count = len(clean_train_df) - valid_count  # Count of False values (invalid rows)

    print(f"Valid rows: {valid_count}")
    print(f"Invalid rows: {invalid_count}")

    base_path = f"/home/stud/ath/ath_ws/datasets/match_april/{threshold}_inline"
    # base_path = f"/home/stud/ath/ath_ws/datasets/match_april/{threshold}"
    
    if os.path.exists(base_path):
        shutil.rmtree(base_path) 
        
    os.makedirs(base_path) 

    clean_train_df.to_csv(f"{base_path}/train.csv", index=False)
    clean_val_df.to_csv(f"{base_path}/val.csv", index=False)
    clean_test_df.to_csv(f"{base_path}/test.csv", index=False)
    
    print_dataset_stats(clean_train_df, clean_val_df, clean_test_df)

    print("Done!")



if __name__ == "__main__":
    main()
