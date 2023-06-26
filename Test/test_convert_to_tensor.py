
import tensorflow as tf

def convert_to_tensor(*arrays):
    """
    Convert input arrays to TensorFlow tensors.

    Parameters:
    - arrays (tuple or list): Input arrays to be converted.

    Returns:
    - converted_tensors (list): List of TensorFlow tensors.
    """

    return [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in arrays]

# Test della funzione convert_to_tensor
converted_tensors = convert_to_tensor(labels, samples)
for tensor in converted_tensors:
    print("Converted tensor shape:", tensor.shape)
