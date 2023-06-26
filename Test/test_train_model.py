from functions import train_step
import numpy as np
import tensorflow as tf
from class_CVAE import CVAE
from configuration_file import conv_architectures, linear_architectures

def test_train_model():
    """
    Test function for the train_model function.

    This function tests the functionality of the train_model function by:
        - Setting the random seed for reproducibility.
        - Creating an instance of CVAE with specified parameters.
        - Creating mock datasets for training, test, and validation.
        - Calling the train_model function to train the model.
        - Asserting that the returned model is an instance of CVAE.
        - Calculating the loss on the validation and test datasets.
        - Asserting that the validation loss is lower than the test loss.

    Raises:
        AssertionError: If any of the assertions fail.
    """
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

    # Create mock datasets for training, test, and validation
    train_dataset = np.random.randn(100, input_dim)
    test_dataset = np.random.randn(100, input_dim)
    val_dataset = np.random.randn(100, input_dim)

    # Call the train_model function
    model = main.train_model(
        latent_dim=latent_dim,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        label_dim=label_dim,
        conv_architectures=conv_architectures,
        linear_architectures=linear_architectures,
        batch_size=32,  # Set your desired batch size here
        input_dim=input_dim,
        epochs=10,  # Set the number of epochs for training
        num_examples_to_generate=6
    )

    # Assert that the returned model is an instance of CVAE
    assert isinstance(model, CVAE)


    # Calculate the loss on the validation dataset
    val_loss = 0
    for val_x in val_dataset:
        val_loss += compute_loss(model, val_x, input_dim)
    val_loss /= len(val_dataset)

    # Calculate the loss on the test dataset
    test_loss = 0
    for test_x in test_dataset:
        test_loss += compute_loss(model, test_x, input_dim)
    test_loss /= len(test_dataset)

    # Assert that the validation loss is lower than the test loss
    assert val_loss < test_loss


# Run the test function
test_train_model()

