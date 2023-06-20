import main
import numpy as np
import tensorflow as tf
from class_CVAE import CVAE, conv_architectures, linear_architectures

def test_reparameterize():
    """
    Test the reparameterize method of the CVAE class in the main module
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

    # Generate random mean and logvar tensors
    batch_size = 64
    mean = np.random.randn(batch_size, latent_dim)
    logvar = np.random.randn(batch_size, latent_dim)

    # Convert the mean and logvar tensors to TensorFlow tensors
    mean = tf.convert_to_tensor(mean, dtype=tf.float32)
    logvar = tf.convert_to_tensor(logvar, dtype=tf.float32)

    # Call the reparameterize method
    z = cvae.reparameterize(mean, logvar)

    # Check the shape of the output tensor
    assert z.shape == (batch_size, latent_dim)

# Run the test function
test_reparameterize()



    
