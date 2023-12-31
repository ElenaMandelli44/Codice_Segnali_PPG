import tensorflow as tf

def create_dataset(data, batch_size, shuffle=True):
    """
    Create a TensorFlow dataset from input data.

    Parameters:
    - data (np.ndarray): Input data array.
    - batch_size (int): The batch size for the dataset.
    - shuffle (bool): Whether to shuffle the data. Default is True.

    Returns:
    - dataset (tf.data.Dataset): The created TensorFlow dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.shuffle(data.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

# Fix a seed for random data generation
np.random.seed(42)

# Testing the create_dataset function
data = np.random.randn(100, 10)
batch_size = 64
dataset = create_dataset(data, batch_size)
print("Created dataset:", dataset)



