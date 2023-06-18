import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random

def test_get_data():
    """
    Test the get_data() function of the main module
    """
    batch_size = 64
    (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        latent_dim,
        label_dim,
        input_dim,
        labels,
    ) = main.get_data(batch_size)

    # Check if the returned datasets are not None
    assert train_dataset is not None
    assert val_dataset is not None
    assert test_dataset is not None

    # Check if the returned input dimensions and labels are of the correct type
    assert isinstance(input_dim, int)
    assert isinstance(latent_dim, int)
    assert isinstance(label_dim, int)
    assert isinstance(labels, list)

def test_train_or_load_model():
    """
    Test the train_or_load_model() function of the main module
    """
    batch_size = 64
    (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        latent_dim,
        label_dim,
        input_dim,
        labels,
    ) = main.get_data(batch_size)

    model = main.train_or_load_model(
        epochs=100,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_architectures=main.conv_architectures,
        linear_architectures=main.linear_architectures,
        batch_size=batch_size,
        input_dim=input_dim,
    )

    # Check if the returned model is an instance of TensorFlow's keras.Model
    assert isinstance(model, keras.Model)


