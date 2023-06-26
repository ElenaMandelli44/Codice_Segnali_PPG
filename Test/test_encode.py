import main
import numpy as np
import tensorflow as tf
import class_CVAE import CVAE, conv_architectures,linear_architectures

def test_encode():
    """
    Test the encode method of the CVAE class.

    This function creates an instance of the CVAE class and tests its encode method.
    It checks if the mean and log variance tensors have the expected shape and type.

    Returns:
        None
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

    # Set the random seed
    tf.random.set_seed(42)
    np.random.seed(42)
    
    x = np.random.randn(batch_size, *input_shape)

    # Convert the input tensor to a TensorFlow tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Call the encode method
    mean, logvar = cvae.encode(x)

    # Check the shapes of mean and logvar
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

    # Check if the mean and logvar tensors are of type tf.Tensor
    assert isinstance(mean, tf.Tensor)
    assert isinstance(logvar, tf.Tensor)
    
    # Check the values of mean and logvar
    #in the test, the mean and logvar tensors are initialized with zeros of the same shape as the expected output.
    #Then, we use np.allclose to compare the tensor values to the expected zeros.
    #If the values are close enough (within a certain tolerance), the assertions will pass. 
    #This is a way to check if the encoding process is working properly and produces output in line with our expectations. 
    #If the mean and logvar tensors are significantly different from zero, it could indicate a problem in the encoding mechanism.
    
    assert np.allclose(mean.numpy(), np.zeros((batch_size, latent_dim)))
    assert np.allclose(logvar.numpy(), np.zeros((batch_size, latent_dim)))









# Run the test function
test_encode()


