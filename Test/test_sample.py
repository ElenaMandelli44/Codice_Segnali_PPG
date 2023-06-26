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
    Test the sample method of the CVAE class.

    This function creates an instance of the CVAE class and tests its sample method. It checks if the
    generated samples have the expected shape and type.

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

    # Set the random seed
    tf.random.set_seed(42)
    np.random.seed(42)

    
    # Set the input parameters for the sample method
    eps = tf.random.normal(shape=(5, latent_dim))
    labels = tf.random.normal(shape=(5, label_dim))
    num_samples = 5

    # Call the sample method
    samples = cvae.sample(eps=eps, labels=labels, num_samples=num_samples)

    # Check the shape of the generated samples
    assert samples.shape == (num_samples, input_dim, 1)
    
    # Check if the generated samples are of type tf.Tensor
    assert isinstance(samples, tf.Tensor)



# Run the test function
test_sample()










