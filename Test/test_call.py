
import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random
import tensorflow as tf

def test_call():
    """
    Test the call method of the CVAE class.

    This function creates an instance of the CVAE class and tests its call method. It checks if the
    forward pass of the CVAE model returns the expected outputs.

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
    cvae = main.CVAE(latent_dim, label_dim, conv_architectures, linear_architectures, input_dim)

    # Define the input data
    batch_size = 64
    input_shape = (batch_size, input_dim, 1)
    label_shape = (batch_size, label_dim)

    # Call the call method
    x_logit, z, mean, logvar = cvae.call(x, y)

    # Check the shape of the output tensors
    assert x_logit.shape == (64, input_dim, 1)
    assert z.shape == (64, latent_dim)
    assert mean.shape == (64, latent_dim)
    assert logvar.shape == (64, latent_dim)

    # Check if the returned tensors are of type tf.Tensor
    assert isinstance(x_logit, tf.Tensor)
    assert isinstance(z, tf.Tensor)
    assert isinstance(mean, tf.Tensor)
    assert isinstance(logvar, tf.Tensor)

# Run the test function
test_call()
