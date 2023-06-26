import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os

def load_data(file_path):
    """
    Load data from a pickle file specified by the file path.
    
    Parameters:
        - file_path: Full path of the pickle file containing the data to load.
    
    Returns:
        A tuple containing:
        - labels: Pandas DataFrame containing the data labels.
        - samples: NumPy array representing the preprocessed data samples.
    """
    with open(file_path, "rb") as file:
        df = pickle.load(file)
    labels = pd.DataFrame(df["labels"])
    samples = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
    samples = np.expand_dims(samples, axis=-1)
    return labels, samples

def combine_data(labels, samples):
    """
    Combine labels and samples to create a single dataset.

    Parameters:
        - labels: NumPy array or Pandas DataFrame representing the data labels.
        - samples: NumPy array representing the data samples.

    Returns:
        A NumPy array representing the combined dataset, where labels are appended to the samples.
    """
    labels = np.expand_dims(labels, axis=-1)
    combined_data = np.hstack([samples, labels])
    return combined_data

def convert_to_tensor(*arrays):
    """
    Convert input arrays to TensorFlow tensors with dtype=tf.float32.

    Parameters:
        - *arrays: Variable number of arrays to be converted to tensors.

    Returns:
        A list of TensorFlow tensors with dtype=tf.float32 representing the converted arrays.
    """
    return [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in arrays]

def create_dataset(data, batch_size, shuffle=True):
    """
    Create a TensorFlow dataset from input data.

    Parameters:
        - data: NumPy array or TensorFlow tensor representing the input data.
        - batch_size: The desired batch size for the created dataset.
        - shuffle: (Optional) Boolean indicating whether to shuffle the data. Default is True.

    Returns:
        A TensorFlow dataset object created from the input data, with batches of size batch_size.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.shuffle(data.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def get_data(batch_size, sampling_rate=100, working_dir=None):
    """
    Load and preprocess data from pickle files and create TensorFlow datasets.

    Parameters:
        - batch_size: The desired batch size for the created datasets.
        - sampling_rate: (Optional) The sampling rate to downsample the data. Default is 100.
        - working_dir: (Optional) The working directory to load the pickle files from. If not provided, the current working directory is used.

    Returns:
        A tuple containing:
        - train_dataset: TensorFlow dataset containing the training data.
        - val_dataset: TensorFlow dataset containing the validation data.
        - test_dataset: TensorFlow dataset containing the test data.
        - input_dim: Dimension of the input signals.
        - latent_dim: Dimension of the target labels.
        - label_dim: Dimension of the target labels.
    """
    if working_dir is None:
        working_dir = os.getcwd()

    train_labels, train_samples = load_data(os.path.join(working_dir, "train_db_1p.pickle"))
    validation_labels, validation_samples = load_data(os.path.join(working_dir, "validation_db_1p.pickle"))
    test_labels, test_samples = load_data(os.path.join(working_dir, "test_db_1p.pickle"))

    x_train = train_samples[::sampling_rate]
    x_val = validation_samples[::sampling_rate]
    x_test = test_samples[::sampling_rate]
    y_train = train_labels[::sampling_rate]
    y_val = validation_labels[::sampling_rate]
    y_test = test_labels[::sampling_rate]

    xy_train = combine_data(y_train, x_train)
    xy_val = combine_data(y_val, x_val)
    xy_test = combine_data(y_test, x_test)

    datasets = convert_to_tensor(xy_train, xy_val, xy_test)
    train_dataset = create_dataset(datasets[0], batch_size)
    val_dataset = create_dataset(datasets[1], batch_size)
    test_dataset = create_dataset(datasets[2], batch_size)

    input_dim = x_train.shape[1]
    latent_dim = y_train.shape[1]
    label_dim = latent_dim

    return train_dataset, val_dataset, test_dataset, input_dim, latent_dim, label_dim, train_labels


