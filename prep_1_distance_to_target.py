import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pprint import pprint
import seaborn as sns


def calculate_target_error(training_df):
    errors = np.sqrt((training_df["x_guess"] - training_df["x1"]) ** 2 +
                     (training_df["y_guess"] - training_df["y1"]) ** 2)
    return errors


def calculate_reference_error(training_df):
    errors = np.sqrt((training_df["x_guess"] - training_df["x0"]) ** 2 +
                     (training_df["y_guess"] - training_df["y0"]) ** 2)
    return errors


def round_dict(stats):
    return {key: round(value, 2) for key, value in stats.items()}


def generate_error_statistics(target_errors, reference_errors, threshold, training_df):
    stats = {
        "mean_target_error": np.mean(target_errors),
        "median_target_error": np.median(target_errors),
        "std_dev_target_error": np.std(target_errors),
        "min_target_error": np.min(target_errors),
        "max_target_error": np.max(target_errors),

        "count_above_threshold_target_error": np.sum(target_errors > threshold),
        "count_above_threshold_x": np.sum(np.abs(training_df["x_guess"] - training_df["x1"]) > threshold),
        "count_above_threshold_y": np.sum(np.abs(training_df["y_guess"] - training_df["y1"]) > threshold),

        "mean_reference_error": np.mean(reference_errors),
        "median_reference_error": np.median(reference_errors),
        "std_dev_reference_error": np.std(reference_errors),
        "min_reference_error": np.min(reference_errors),
        "max_reference_error": np.max(reference_errors),
    }
    return round_dict(stats)


def compute_estimate_accuracy_percentages(target_errors):
    thresholds = [90, 60, 30, 25, 5, 2, 1.5, 1.25, 1]
    percentages = {f"accuracy_within_{t}_pixels": np.mean(target_errors <= t) * 100 for t in thresholds}
    return round_dict(percentages)


def save_high_error_data(target_errors, threshold, training_df):
    high_error_rows = training_df.loc[target_errors > threshold, :]
    high_error_rows.to_csv("/home/stud/ath/ath_ws/datasets/match_april/high_error_kpids.csv", index=False)
    return high_error_rows


def count_errors_per_dataset(high_error_rows):
    return high_error_rows["dataset"].value_counts()


def analyze_error_correlation(reference_errors, target_errors):
    pearson_corr, pearson_p = pearsonr(reference_errors, target_errors)
    spearman_corr, spearman_p = spearmanr(reference_errors, target_errors)
    return {
        "pearson_correlation": round(pearson_corr, 2),
        "pearson_p_value": round(pearson_p, 5),
        "spearman_correlation": round(spearman_corr, 2),
        "spearman_p_value": round(spearman_p, 5),
    }


def plot_error_relationship(reference_errors, target_errors):
    plt.figure(figsize=(8, 6))
    plt.scatter(reference_errors, target_errors, alpha=0.5, s=10)
    plt.xlabel("Reference Error (Distance to x0, y0)")
    plt.ylabel("Target Error (Distance to x1, y1)")
    plt.title("Reference Error vs. Target Error")
    plt.grid(True)
    plt.show()


def remove_outliers(data, threshold=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    return data[(data >= lower) & (data <= upper)]


def plot_histograms(target_errors, reference_errors, remove_outliers_flag=False):
    target_plot = remove_outliers(np.array(target_errors)) if remove_outliers_flag else target_errors
    reference_plot = remove_outliers(np.array(reference_errors)) if remove_outliers_flag else reference_errors

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(target_plot, bins=50, alpha=0.7, color='b')
    plt.title("Histogram of Target Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(reference_plot, bins=50, alpha=0.7, color='r')
    plt.title("Histogram of Reference Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    
    plt.show()


def plot_dataset_error_counts(dataset_counts):
    dataset_counts.plot(kind='bar', figsize=(15, 5), color='purple')
    plt.xlabel("Dataset")
    plt.ylabel("High Error Count")
    plt.title("High Error Counts per Dataset")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()


def main(threshold):
    training_df = pd.read_csv(
        "/home/stud/ath/ath_ws/datasets/match_april/training_may_mo_mg.csv",
        header=0,
        names=(
            "dataset", "cam", "kpid", "pair_name", "x0", "y0", "x1", "y1", "x_guess", "y_guess", "certainty",
        )
    )
    
    # training_mg_df = pd.read_csv(
    #     "/home/stud/ath/ath_ws/datasets/match_april/training_may_mo_mg.csv",
    #     header=0,
    #     names=(
    #         "dataset", "cam", "kpid", "pair_name", "x0", "y0", "x1", "y1", "x_guess", "y_guess", "certainty",
    #     )
    # )
    
    # training_df = pd.concat([training_df, training_mg_df], ignore_index=True)
    
    target_errors = calculate_target_error(training_df)
    reference_errors = calculate_reference_error(training_df)

    stats = generate_error_statistics(target_errors, reference_errors, threshold, training_df)
    print("=> Error statistics:")
    pprint(stats)

    accuracy_percentages = compute_estimate_accuracy_percentages(target_errors)
    print("=> Estimate Accuracy percentages:")
    for key, value in accuracy_percentages.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}%")

    high_error_rows = save_high_error_data(target_errors, threshold, training_df)
    dataset_counts = count_errors_per_dataset(high_error_rows)
    print("=> Number of high-error rows per dataset:")
    print(dataset_counts.to_string())

    correlation_results = analyze_error_correlation(reference_errors, target_errors)
    print('=> Correlation results:')
    pprint(correlation_results)

    plot_error_relationship(reference_errors, target_errors)
    plot_histograms(target_errors, reference_errors, remove_outliers_flag=True)
    plot_dataset_error_counts(dataset_counts)

    print('Done!')


if __name__ == "__main__":
    main()
