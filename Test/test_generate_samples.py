import main
import numpy as np
import tensorflow as tf
from class_CVAE import CVAE
from configuration_file import conv_architectures, linear_architectures

def test_generate_samples():
    """
    Test the generate_samples function in the main module
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
    batch_size = 32
    sample = tf.random.normal(shape=(batch_size, input_dim + label_dim, 1))
    n = 5

    # Call the generate_samples function
    result_x, result_y = main.generate_samples(cvae, sample, n, input_dim)

    # Check if the generated samples have the correct shapes
    expected_shape_x = (n, input_dim, 1)
    expected_shape_y = (n, latent_dim)
    assert result_x.shape == expected_shape_x
    assert result_y.shape == expected_shape_y

    # Check if the generated samples are valid predictions
    predictions = cvae.decode(tf.convert_to_tensor(result_x, dtype=tf.float32))
    assert predictions.shape == expected_shape_x

# Run the test function
test_generate_samples()





 

