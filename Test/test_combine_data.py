
import numpy as np

def combine_data(labels, samples):
    """
    Combine labels and samples into a single array.

    Parameters:
    - labels (np.ndarray): The labels associated with the samples.
    - samples (np.ndarray): The samples data.

    Returns:
    - combined_data (np.ndarray): The combined data array.
    """

    labels = np.expand_dims(labels, axis=-1)
    combined_data = np.hstack([samples, labels])
    return combined_data

# Test della funzione combine_data
combined_data = combine_data(labels, samples)
print("Combined data shape:", combined_data.shape)
