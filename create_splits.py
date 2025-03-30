import pandas as pd


def filter_high_error_kpids(training_df, high_error_kpids_df):
    """Filter training data by removing rows corresponding to high error keypoints."""
    high_error_kpids = high_error_kpids_df["kpid"].values
    clean_df = training_df[~training_df["kpid"].isin(high_error_kpids)]
    return clean_df


def create_datasets(df, splits):
    """Create clean training, validation, and test datasets based on the splits dictionary."""
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


def main():
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

    training_df = pd.read_csv("data/training.csv")
    high_error_kpids_df = pd.read_csv("data/high_error_kpids.csv")

    # Step 1: Filter out high error keypoints from the training data
    clean_df = filter_high_error_kpids(training_df, high_error_kpids_df)

    # Step 2: Create clean datasets for train, val, and test splits
    clean_train_df, clean_val_df, clean_test_df = create_datasets(clean_df, splits)

    # Step 3: Save the cleaned datasets to CSV
    clean_train_df.to_csv("data/train.csv", index=False)
    clean_val_df.to_csv("data/val.csv", index=False)
    clean_test_df.to_csv("data/test.csv", index=False)

    # Step 4: Print the dataset stats
    print_dataset_stats(clean_train_df, clean_val_df, clean_test_df)

    print('Done!')


if __name__ == "__main__":
    main()
