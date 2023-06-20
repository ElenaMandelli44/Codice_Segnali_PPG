import main
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from generate_signals import generate_and_save_images
from class_CVAE import CVAE, conv_architectures, linear_architectures
import pandas as pd

def test_generate_and_save_images():
    """
    Test the generate_and_save_images function in the main module
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


    # Generate random test sample
    test_sample = tf.random.normal(shape=(10, cvae.input_dim + cvae.label_dim, 1))

    # Create an instance of CVAE
    cvae = CVAE(latent_dim=10, label_dim=10, conv_architectures=[], linear_architectures=[], input_dim=10)

    # Call the generate_and_save_images function
    main.generate_and_save_images(cvae, epoch=1, test_sample=test_sample, input_dim=cvae.input_dim)

    # Check if the plot is displayed without errors
    assert plt.gcf() is not None

# Run the test function
test_generate_and_save_images()

