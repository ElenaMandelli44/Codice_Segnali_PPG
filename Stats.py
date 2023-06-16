import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm


def compute_metrics(y_true, y_pred):
    # Coefficient of Determination (R²)
    r2 = r2_score(y_true, y_pred)
    # print("Coefficient of Determination (R²):", r2)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse


def analyze():
    """
    Perform analysis on test and predicted data.

    This function performs the following steps:
    1. Sets the working directory.
    2. Loads test data from a pickle file.
    3. Normalizes the test data.
    4. Retrieves the predicted signals and labels from a pickle file.
    5. Normalizes the predicted signals.
    6. Determines the unique ages present in the predicted data that also exist in the test data.
    7. Computes metrics (R2 and MSE) for each age, comparing the test and generated signals.
    8. Plots the computed metrics (R2 and MSE) against the ages.

    """

   

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
                r2, mse = compute_metrics(test_age_signal, gen_age_signal)
                all_metrics.append([r2, mse])

        all_metrics_array = np.asarray(all_metrics).mean(axis=0)
        if np.isnan(all_metrics_array).any():
            ipdb.set_trace()
        metrics_per_age.append(all_metrics_array)

    metrics_per_age = np.asarray(metrics_per_age)

    _, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(unique_ages, metrics_per_age[:, 0], "o", label="R2")
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel("R2")
    ax[0].legend()
    ax[1].plot(unique_ages, metrics_per_age[:, 1], "o", label="MSE")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("MSE")
    ax[1].legend()
    plt.show()

