import pickle
import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load data from a pickle file.

    Parameters:
      - file_path (str): The path to the pickle file.

    Returns:
      - labels (pd.DataFrame): The labels associated with the samples.
      - samples (np.ndarray): The samples data, normalized and expanded with an additional dimension.
    """

    with open(file_path, "rb") as file:
        df = pickle.load(file)
    labels = pd.DataFrame(df["labels"])
    samples = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
    samples = np.expand_dims(samples, axis=-1)
    return labels, samples

# Test della funzione load_data
file_path = "data.pickle"
labels, samples = load_data(file_path)
print("Loaded labels shape:", labels.shape)
print("Loaded samples shape:", samples.shape)
