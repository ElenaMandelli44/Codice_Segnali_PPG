import main
import numpy as np
import tensorflow as tf
from functions import log_normal_pdf, compute_loss
from class_CVAE import CVAE
from configuration_file import conv_architectures, linear_architectures

def test_log_normal_pdf():
    """
    Test function for the log_normal_pdf function in the main module.

    This function tests the functionality of the log_normal_pdf function by:
        - Defining the input tensors for the first test case.
        - Calling the log_normal_pdf function with the defined input tensors.
        - Checking the shape and values of the result.
        - Defining the input parameters for CVAE initialization for the second test case.
        - Creating an instance of CVAE.
        - Generating random input tensors.
        - Calling the log_normal_pdf function with the generated input tensors.
        - Checking the shape of the output tensor.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    #First Test
    
    # Define the input tensors (direct)
    sample = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mean = tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    logvar = tf.constant([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    # Call the log_normal_pdf function
    result = main.log_normal_pdf(sample, mean, logvar)

    # Check if the result has the correct shape
    assert result.shape == (2,)

    # Check the values of the result
    expected_result = np.array([-3.081085, -9.150937])
    np.testing.assert_allclose(result.numpy(), expected_result, rtol=1e-6)

    #Second Test (indirect)
    
    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    input_dim = 1024
    tf.random.set_seed(42)
    
    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    
    # Generate random input tensors
    batch_size = 64
    sample = tf.random.normal(shape=(batch_size, 10))
    mean = tf.random.normal(shape=(batch_size, 10))
    logvar = tf.random.normal(shape=(batch_size, 10))

    # Call the log_normal_pdf function
    result = main.log_normal_pdf(sample, mean, logvar)

    # Check if the output tensor has the correct shape
    expected_shape = (batch_size,)
    assert result.shape == expected_shape


def test_compute_loss():
    """
    Test function for the compute_loss function in the main module.

    This function tests the functionality of the compute_loss function by:
        - Creating an instance of CVAE.
        - Generating random input tensors for two test cases.
        - Calling the compute_loss function with the generated input tensors.
        - Checking if the output loss is a scalar value.
        - Checking if the loss for all zeros input is smaller than the loss for random input.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    # Test case 1:Generate random input tensor
    batch_size = 64
    # Set the random seed
    tf.random.set_seed(42)
    
    x = tf.random.normal(shape=(batch_size, cvae.input_dim + cvae.label_dim, 1))

    # Call the compute_loss function
    loss = main.compute_loss(cvae, x, cvae.input_dim)

    # Check if the loss is a scalar value
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()

    # Test case 2: All zeros input tensor
    x_zeros = tf.zeros_like(x)

    # Call the compute_loss function
    loss_zeros = main.compute_loss(cvae, x_zeros, cvae.input_dim)

    # Check if the loss is a scalar value
    assert isinstance(loss_zeros, tf.Tensor)
    assert loss_zeros.shape == ()

    # Check if the loss for all zeros input is smaller than the loss for random input
    assert loss_zeros < loss

# Run the test functions
test_log_normal_pdf()
test_compute_loss()








