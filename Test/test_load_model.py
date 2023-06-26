import numpy as np
import tensorflow as tf
from class_CVAE import CVAE
from configuration_file import conv_architectures, linear_architectures

def test_load_model():
    # Set the random seed
    np.random.seed(42)
    tf.random.set_seed(42)

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

    # Save the model weights
    model.save_weights("trained_model")

    # Call the load_model function
    loaded_model = main.load_model(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_architectures=conv_architectures,
        linear_architectures=linear_architectures,
        input_dim=input_dim,
    )

    # Assert that the returned model is an instance of CVAE
    assert isinstance(loaded_model, CVAE)

    # Assert that the loaded model has the same architecture as the original model
    assert loaded_model.latent_dim == cvae.latent_dim
    assert loaded_model.label_dim == cvae.label_dim
    assert loaded_model.conv_layers_settings == cvae.conv_layers_settings
    assert loaded_model.linear_layers_settings == cvae.linear_layers_settings
    assert loaded_model.input_dim == cvae.input_dim

    # Delete the saved model weights
    os.remove("trained_model")

# Run the test function
test_load_model()

