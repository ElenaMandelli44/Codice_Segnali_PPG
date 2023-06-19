import main
import numpy as np
import tensorflow as tf

def test_reparameterize():
    """
    Test the reparameterize method of the CVAE class in the main module
    """
    # Create an instance of CVAE
    latent_dim = 10
    label_dim = 10
    conv_architectures = [64, 128, 256]
    linear_architectures = [256, 128]
    input_dim = (1024, 1)
    cvae = main.CVAE(latent_dim, label_dim, conv_architectures, linear_architectures, input_dim)

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

