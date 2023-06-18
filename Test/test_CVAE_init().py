import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random

def test_CVAE_init():
    """
    Test the __init__ method of the CVAE class in the main module
    """
    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    conv_architectures = [64, 128, 256]
    linear_architectures = [256, 128]
    input_dim = 1024

    # Create an instance of CVAE
    cvae = main.CVAE(latent_dim, label_dim, conv_architectures, linear_architectures, input_dim)

    # Check if the attributes are correctly set
    assert cvae.latent_dim == latent_dim
    assert cvae.label_dim == label_dim
    assert cvae.input_dim == input_dim

    # Check if the encoder and decoder are instances of Keras models
    assert isinstance(cvae.encoder, keras.Model)
    assert isinstance(cvae.decoder, keras.Model)

# Run the test function
test_CVAE_init()

