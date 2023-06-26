import main
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import pickle
from generate_signals import generate_samples_from_age
from class_CVAE import CVAE, conv_architectures, linear_architectures


def test_generate_samples_from_age():
    """
    Test the generate_samples_from_age function in the main module
    """

    # Generate random train_labels and age
    train_labels = pd.DataFrame(np.random.randn(10, 10))
    # Test case 1: age = 30, n = 5
    age = 30
    n = 5
    
    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    # conv_architectures = [64, 128, 256]
    # linear_architectures = [256, 128]
    input_dim = 1024

    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    # Call the generate_samples_from_age function
    result_x, result_y = generate_samples_from_age(cvae, train_labels, age, n)

    # Check the shapes of the generated samples
    assert result_x.shape == (n, cvae.input_dim)
    assert result_y.shape == (n, cvae.label_dim + 1)

    # Test case 2: age = 40, n = 3
    age = 40
    n = 3
    
    # Call the generate_samples_from_age function again
    result_x, result_y = generate_samples_from_age(cvae, train_labels, age, n)

    # Check the shapes of the generated samples
    assert result_x.shape == (n, cvae.input_dim)
    assert result_y.shape == (n, cvae.label_dim + 1)


# Run the test function
test_generate_samples_from_age()


