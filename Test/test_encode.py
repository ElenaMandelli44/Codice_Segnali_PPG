import main
import numpy as np
import tensorflow as tf
import class_CVAE import CVAE, conv_architectures,linear_architectures

def test_encode():
    """
    Test the encode method of the CVAE class in the main module
    """

    #Define the input parameters for CVAE initialization
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

    # Generate a random input tensor
    batch_size = 64
    input_shape = (1024, 1)
    x = np.random.randn(batch_size, *input_shape)

    # Convert the input tensor to a TensorFlow tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Call the encode method
    mean, logvar = cvae.encode(x)

    # Check the shapes of mean and logvar
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

# Run the test function
test_encode()


