import main
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def test_generate_and_save_images():
    """
    Test the generate_and_save_images function in the main module
    """
    # Create an instance of CVAE
    class CVAE:
        def __init__(self):
            self.input_dim = 10
            self.label_dim = 10

    class CVAE(tf.keras.Model):
        def __init__(self, latent_dim, label_dim, conv_architectures, linear_architectures, input_dim):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.label_dim = label_dim
            self.input_dim = input_dim

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

