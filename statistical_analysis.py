import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.stats import spearmanr

def compute_metrics(y_true, y_pred):

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return spearman_corr, mse


def analyze():
    # Definition of working directory
    working_dir = ""

    # Loading test data
    with open(working_dir + "test_db_1p.pickle", "rb") as file:
        df = pickle.load(file)

    test = np.asarray([d / np.max(np.abs(d)) for d in df["samples"]])
    test = np.expand_dims(test, axis=-1)
    test_labels = np.asarray(df["labels"]).astype(np.float32)

    # Generated signals
    with open("generated_samples_18_992.pickle", "rb") as file:
        predicted_signal, predicted_signal_labels = pickle.load(file)
    gen = np.asarray([d / np.max(np.abs(d)) for d in predicted_signal])
    gen = gen[..., None]

    unique_predicted_ages = np.sort(np.unique(predicted_signal_labels[:, 1]))
    unique_ages = [
        int(age) for age in unique_predicted_ages if age in test_labels[:, 1]
    ]
    metrics_per_age = []
    for age in tqdm(unique_ages):
        test_age_indices = np.where(test_labels[:, 1] == age)[0]
        gen_age_indices = np.where(predicted_signal_labels[:, 1] == age)[0]

        all_metrics = []
        for test_age_idx in test_age_indices:
            test_age_signal = test[test_age_idx]
            for gen_age_idx in gen_age_indices:
                gen_age_signal = gen[gen_age_idx]
                spearman_corr , mse = compute_metrics(test_age_signal, gen_age_signal)
                all_metrics.append([spearman_corr, mse])

        all_metrics_array = np.asarray(all_metrics).mean(axis=0)
        if np.isnan(all_metrics_array).any():
            ipdb.set_trace()
        metrics_per_age.append(all_metrics_array)

    metrics_per_age = np.asarray(metrics_per_age)

    _, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(unique_ages, metrics_per_age[:, 0], "o", label="Spearman Corr")
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel("")
    ax[0].legend()
    ax[1].plot(unique_ages, metrics_per_age[:, 1], "o", label="MSE")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("MSE")
    ax[1].legend()
    plt.show()
    
    
    
    # Calculate and display the average and median of MSE and Spearman correlation
    mse_average = np.average(metrics_per_age[:, 0])
    mse_median = np.median(metrics_per_age[:, 0])
    spearman_average = np.average(metrics_per_age[:, 1])
    spearman_median = np.median(metrics_per_age[:, 1])
    print(f"Average MSE: {mse_average}")
    print(f"Median MSE: {mse_median}")
    print(f"Average Spearman Corr: {spearman_average}")
    print(f"Median Spearman Corr: {spearman_median}")

    plt.show()


if __name__ == "__main__":
    analyze()

