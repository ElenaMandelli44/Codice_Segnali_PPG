import main
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import pickle
from generate_signals import generate_samples_from_age
from class_CVAE import CVAE, conv_architectures, linear_architectures


def test_generate_samples_from_age():
    """
    Test the generate_samples_from_age function in the main module
    """

    # Generate random train_labels and age
    train_labels = pd.DataFrame(np.random.randn(10, 10))
    age = 30
    n = 5

    # Create an instance of CVAE
    cvae = CVAE()

    # Call the generate_samples_from_age function
    result_x, result_y = main.generate_samples_from_age(cvae, train_labels, age, n)

    # Check the shapes of the generated samples
    assert result_x.shape == (n, cvae.input_dim + cvae.label_dim, 1)
    assert result_y.shape == (n, cvae.label_dim)

def test_save_samples_from_age_range():
    """
    Test the save_samples_from_age_range function in the main module
    """
    # Create an instance of CVAE
    class CVAE:
        def __init__(self):
            self.latent_dim = 10
            self.label_dim = 10

    # Generate random train_labels, min_age, max_age, and n
    train_labels = pd.DataFrame(np.random.randn(10, 10))
    min_age = 20
    max_age = 30
    n = 5

    # Create an instance of CVAE
    cvae = CVAE()

    # Call the save_samples_from_age_range function
    main.save_samples_from_age_range(cvae, train_labels, min_age, max_age, n)

    # Check if the file is saved successfully
    filename = f"generated_samples_{min_age}_{max_age}9.pickle"
    assert os.path.exists(filename)

    # Delete the created file
    os.remove(filename)

# Run the test functions
test_generate_samples_from_age()
test_save_samples_from_age_range()






 

