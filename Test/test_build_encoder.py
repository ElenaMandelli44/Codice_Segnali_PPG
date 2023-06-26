import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random
import tensorflow as tf
import class_CVAE import CVAE

def test_build_encoder():
    """
    Test the build_encoder method of the CVAE class in the main module.

    This function creates an instance of the CVAE class and tests its build_encoder method. It checks
    if the generated encoder model has the correct architecture, input shape, and output shape.

    Returns:
        None
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


    # Call the build_encoder method
    encoder = cvae.build_encoder(conv_architectures, linear_architectures)

    # Check if the encoder is an instance of Keras Sequential model
    assert isinstance(encoder, tf.keras.Sequential)

    # Check if the number of layers in the encoder is correct
    assert len(encoder.layers) == len(conv_architectures) + len(linear_architectures) + 4

    # Check the input shape of the encoder
    assert encoder.input_shape == (None, cvae.input_dim, 1)

    # Check the output shape of the encoder
    assert encoder.output_shape == (None, 2 * cvae.latent_dim)




# Run the test function
test_build_encoder()
