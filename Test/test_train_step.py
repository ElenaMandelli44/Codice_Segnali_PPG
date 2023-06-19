import main
import numpy as np
import tensorflow as tf

def test_train_step():
    """
    Test the train_step function in the main module
    """
    # Create an instance of CVAE
    cvae = CVAE(latent_dim=10, label_dim=10, conv_architectures=[], linear_architectures=[], input_dim=10)

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
