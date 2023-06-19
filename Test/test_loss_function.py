import main
import numpy as np
import tensorflow as tf

def test_log_normal_pdf():
    """
    Test the log_normal_pdf function in the main module
    """
    # Generate random input tensors
    batch_size = 32
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
    Test the compute_loss function in the main module
    """
    # Create an instance of CVAE
    class CVAE:
        def __init__(self):
            self.input_dim = 10
            self.label_dim = 5

        def __call__(self, x, y):
            x_logit = tf.random.normal(shape=(x.shape[0], self.input_dim, 1))
            z = tf.random.normal(shape=(x.shape[0], self.label_dim))
            mean = tf.random.normal(shape=(x.shape[0], self.label_dim))
            logvar = tf.random.normal(shape=(x.shape[0], self.label_dim))
            return x_logit, z, mean, logvar

    cvae = CVAE()

    # Generate random input tensor
    batch_size = 32
    x = tf.random.normal(shape=(batch_size, cvae.input_dim + cvae.label_dim, 1))

    # Call the compute_loss function
    loss = main.compute_loss(cvae, x, cvae.input_dim)

    # Check if the loss is a scalar value
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()

# Run the test functions
test_log_normal_pdf()
test_compute_loss()

