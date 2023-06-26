import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random
import tensorflow as tf
from class_CVAE import CVAE

def test_build_decoder():
    """
    Build the decoder model based on the provided convolutional and linear architectures.

    Args:
        conv_architectures (list): List of dictionaries containing the settings for each convolutional layer.
        linear_architectures (list): List of dictionaries containing the settings for each linear layer.

    Returns:
        tf.keras.Sequential: The constructed decoder model.
    """

    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    conv_architectures = [
        [
            {"filters": 64, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"activation": "relu"},
        ],
        [
            {"filters": 128, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"activation": "relu"},
        ],
        [
            {"filters": 256, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"activation": "relu"},
        ],
    ]
    linear_architectures = [
        [{"units": 256, "activation": "relu"}],
        [{"units": 128, "activation": "relu"}],
    ]
    input_dim = 1024

    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    # Call the build_decoder method
    decoder = cvae.build_decoder(conv_architectures, linear_architectures)

    # Check if the decoder is an instance of Keras Sequential model
    assert isinstance(decoder, tf.keras.Sequential)

    # Check if the number of layers in the decoder is correct
    assert len(decoder.layers) == len(conv_architectures) + len(linear_architectures) + 7

    # Define a random input tensor
    # Set the random seed
    tf.random.set_seed(42)
    np.random.seed(42)
    input_tensor = tf.random.normal(shape=(1, latent_dim + label_dim))

    # Pass the input tensor through the decoder
    output_tensor = decoder(input_tensor)

    # Check the shape of the output tensor
    expected_shape = (1, 1024)
    assert output_tensor.shape == expected_shape

    # Check if the output tensor contains finite values
    assert np.all(np.isfinite(output_tensor.numpy()))

# Run the test function
test_build_decoder()

