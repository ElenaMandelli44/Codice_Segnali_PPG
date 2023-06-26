import os
import tensorflow as tf

def get_data(batch_size, sampling_rate=100, working_dir=None):
    """
    Load and preprocess data for a PPG project and return necessary data and parameters.

    Parameters:
    - batch_size (int): The desired batch size for training the model.
    - sampling_rate (int): The sampling rate for selecting data. Default is 100.
    - working_dir (str): The working directory path. If None, the current working directory will be used.

    Returns:
    - train_dataset (tf.data.Dataset): TensorFlow dataset containing the training data.
    - val_dataset (tf.data.Dataset): TensorFlow dataset containing the validation data.
    - test_dataset (tf.data.Dataset): TensorFlow dataset containing the test data.
    - input_dim (int): Dimension of the input signals.
    - latent_dim (int): Dimension of the target labels.
    - label_dim (int): Dimension of the target labels.
    """

    if working_dir is None:
        working_dir = os.getcwd()

    # Load data
    train_labels, train_samples = load_data(os.path.join(working_dir, "train_db_1p.pickle"))
    validation_labels, validation_samples = load_data(os.path.join(working_dir, "validation_db_1p.pickle"))
    test_labels, test_samples = load_data(os.path.join(working_dir, "test_db_1p.pickle"))

    # Select data based on sampling rate
    x_train = train_samples[::sampling_rate]
    x_val = validation_samples[::sampling_rate]
    x_test = test_samples[::sampling_rate]
    y_train = train_labels[::sampling_rate]
    y_val = validation_labels[::sampling_rate]
    y_test = test_labels[::sampling_rate]

    # Combine data
    xy_train = combine_data(y_train, x_train)
    xy_val = combine_data(y_val, x_val)
    xy_test = combine_data(y_test, x_test)

    # Convert to TensorFlow tensors
    datasets = convert_to_tensor(xy_train, xy_val, xy_test)

    # Create datasets
    train_dataset = create_dataset(datasets[0], batch_size)
    val_dataset = create_dataset(datasets[1], batch_size)
    test_dataset = create_dataset(datasets[2], batch_size)

    # Get dimensions
    input_dim = x_train.shape[1]
    latent_dim = y_train.shape[1]
    label_dim = latent_dim

    return train_dataset, val_dataset, test_dataset, input_dim, latent_dim, label_dim, train_labels


# Testing the get_data function
batch_size = 32
sampling_rate = 50
working_dir = ""
train_dataset, val_dataset, test_dataset, input_dim, latent_dim, label_dim, labels = get_data(batch_size, sampling_rate, working_dir)
print("Train dataset:", train_dataset)
print("Validation dataset:", val_dataset)
print("Test dataset:", test_dataset)
print("Input dimension:", input_dim)
print("Latent dimension:", latent_dim)
print("Label dimension:", label_dim)



