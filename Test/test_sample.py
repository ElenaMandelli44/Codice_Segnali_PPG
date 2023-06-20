import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random
from class_CVAE import CVAE, conv_architectures, linear_architectures
import tensorflow as tf


def test_sample():
    """
    Test the sample method of the CVAE class in the main module
    """
    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    input_dim = 1024
    
    # Create an instance of CVAE
    latent_dim = 10
    label_dim = 10
    conv_architectures = [64, 128, 256]
    linear_architectures = [256, 128]
    input_dim = 1024
    cvae = main.CVAE(latent_dim, label_dim, conv_architectures, linear_architectures, input_dim)

    # Set the input parameters for the sample method
    eps = tf.random.normal(shape=(5, latent_dim))
    labels = tf.random.normal(shape=(5, label_dim))
    num_samples = 5

    # Call the sample method
    samples = cvae.sample(eps=eps, labels=labels, num_samples=num_samples)

    # Check the shape of the generated samples
    assert samples.shape == (5, input_dim, 1)

# Run the test function
test_sample()







