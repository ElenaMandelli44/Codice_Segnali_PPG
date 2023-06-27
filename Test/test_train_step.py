import main
import numpy as np
import tensorflow as tf
from functions import train_step
from class_CVAE import CVAE
from configuration_file import conv_architectures, linear_architectures

def test_train_step():
    """
    Test the train_step function in the main module.

    This function tests the train_step function in the main module by performing the following steps:
    1. Initializes the input parameters for CVAE (Conditional Variational Autoencoder).
    2. Creates an instance of CVAE with the specified parameters.
    3. Generates a random input tensor.
    4. Creates an instance of the Adam optimizer.
    5. Calls the train_step function with the CVAE, input tensor, optimizer, and input dimension.
    6. Checks if the returned loss value is a scalar tensor.

    Returns:
        None

    Raises:
        AssertionError: If the loss value is not a scalar tensor or if the shape of the loss tensor is not ().

    Example usage:
        test_train_step()
    """
    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    input_dim = 1024
    
    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim = latent_dim,
        label_dim = label_dim,
        conv_layers_settings = conv_architectures, 
        linear_architectures= linear_architectures, 
        input_dim= input_dim)

    # Generate random input tensor
    batch_size = 64
    x = tf.random.normal(shape=(batch_size, cvae.input_dim + cvae.label_dim, 1))

    # Create an instance of Adam optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Call the train_step function
    loss = main.train_step(cvae, x, optimizer, cvae.input_dim)

    # Check if the loss is a scalar value
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()

# Run the test function
test_train_step()


