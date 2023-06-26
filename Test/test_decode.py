import main
import numpy as np
import tensorflow as tf
from class_CVAE import CVAE, linear_architectures, conv_architectures

def test_decode():
of the CVAE class in the main module
    """
    Test the decode method of the CVAE class.

    This function creates an instance of the CVAE class and tests its decode method.
    It checks if the reconstructed outputs have the expected shape and type.

    Returns:
        None
    """
    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    input_dim = 1024

    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    # Generate random input tensors
    batch_size = 64

    # Set the random seed
    tf.random.set_seed(42)
    np.random.seed(42)

    z = tf.random.normal(shape=(batch_size, cvae.latent_dim))
    labels = tf.random.normal(shape=(batch_size, cvae.label_dim))

    # Call the decode method
    decoded_output = cvae.decode(z, labels, apply_sigmoid=True)

    # Check if the output tensor has the correct shape
    expected_shape = (batch_size, cvae.input_dim)
    assert decoded_output.shape == expected_shape

    # Check if the sigmoid activation is applied when apply_sigmoid is True
    assert tf.reduce_all(tf.math.logical_and(decoded_output >= 0, decoded_output <= 1))

# Run the test function
test_decode()

