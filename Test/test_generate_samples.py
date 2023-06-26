import main
import numpy as np
import tensorflow as tf
from class_CVAE import CVAE
from configuration_file import conv_architectures, linear_architectures

def test_generate_samples():
    """
    Test function for the generate_samples function in the main module.

    This function tests the functionality of the generate_samples function by:
        - Defining the input parameters for CVAE initialization.
        - Creating an instance of CVAE.
        - Generating a random input tensor.
        - Testing two test cases:
            - Test case 1:
                - Setting the random seed.
                - Calling the generate_samples function with the defined input tensors and n = 5.
                - Checking the shape of the generated samples.
                - Checking if the generated samples are valid predictions.
            - Test case 2:
                - Setting n = 1.
                - Calling the generate_samples function with the defined input tensors and n = 1.
                - Checking the shape of the generated samples.

    Raises:
        AssertionError: If any of the assertions fail.
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

    # Generate random input tensor
    batch_size = 64
    sample = tf.random.normal(shape=(batch_size, input_dim + label_dim, 1))
    n = 5

    #Test case 1: Call the generate_samples function
    # Set the random seed
    tf.random.set_seed(42)
    result_x, result_y = main.generate_samples(cvae, sample, n, input_dim)

    # Check if the generated samples have the correct shapes
    expected_shape_x = (n, input_dim, 1)
    expected_shape_y = (n, latent_dim)
    assert result_x.shape == expected_shape_x
    assert result_y.shape == expected_shape_y

    # Check if the generated samples are valid predictions
    predictions = cvae.decode(tf.convert_to_tensor(result_x, dtype=tf.float32))
    assert predictions.shape == expected_shape_x


    # Test case 2: n = 1
    n = 1
    result_x, result_y = main.generate_samples(cvae, sample, n, input_dim)

    # Check if the generated samples have the correct shapes
    expected_shape_x = (n, input_dim, 1)
    expected_shape_y = (n, latent_dim)
    assert result_x.shape == expected_shape_x
    assert result_y.shape == expected_shape_y


# Run the test function
test_generate_samples()





 

