
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import ipdb
import time
import pickle
import pandas as pd
import random
from IPython import display
from itertools import product
import os
from scipy.stats import spearmanr
from class_CVAE import CVAE


def get_data(batch_size):
    """
        This script is designed to load and preprocess data for a PPG (Photoplethysmography) project. 
        It provides a function called get_data that returns the necessary data and parameters for training a model.

        Parameters:
        - batch_size: The desired batch size for training the model.

        Returns:
        - train_dataset: TensorFlow dataset containing the training data.
        - val_dataset: TensorFlow dataset containing the validation data.
        - test_dataset: TensorFlow dataset containing the test data.
        - input_dim: Dimension of the input signals.
        - latent_dim: Dimension of the target labels.
        - label_dim: Dimension of the target labels.
        - input_dim: Dimension of the input signals.
    """

    
    # Definition of working directory
    working_dir = "/Users/elenamandelli/Desktop/PPG_Project/"

    # Loading training data
    with open(working_dir + "train_db_1p.pickle", "rb") as file:
        df = pickle.load(file)  # DataFrame that contains the uploaded data

    train_labels = pd.DataFrame(df["labels"])

    train = np.asarray(
        [d / np.max(np.abs(d)) for d in df["samples"]]
    )  # Training signlas are normalized by didviding each signal by the absolute maximum value among all signlas.
    train = np.expand_dims(train, axis=-1)

    # Loading validation data
    with open(working_dir + "validation_db_1p.pickle", "rb") as file:
        df = pickle.load(file)

    validation_labels = pd.DataFrame(df["labels"])
    validation = np.asarray([d / np.max(np.abs(d)) for d in df["samples"]])
    validation = np.expand_dims(validation, axis=-1)

    # Loading test data
    with open(working_dir + "test_db_1p.pickle", "rb") as file:
        df = pickle.load(file)

    test_labels = pd.DataFrame(df["labels"])
    raw_test_labels = pd.DataFrame(df["labels"]).values.astype(np.float32)

    test = np.asarray([d / np.max(np.abs(d)) for d in df["samples"]])
    test = np.expand_dims(test, axis=-1)

    # Definition of data for training, validation, and testing.

    """  
    Due to the large number of files, it was decided to work om a reduced number of signlas 
    """

    x_train = train[::100]
    x_test = test[::100]
    x_val = validation[::100]
    y_train = train_labels[::100]
    y_test = test_labels[::100]
    y_val = validation_labels[::100]

    # Expand dimention labels

    """ Expanding the dimensions is mandatory to be able to combine the labels 
            with the features during the training phase """

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)

    # Combine
    xy_train = np.hstack([x_train, y_train])
    xy_val = np.hstack([x_val, y_val])
    xy_test = np.hstack([x_test, y_test])

    input_dim = x_train.shape[1]
    latent_dim = y_train.shape[1]
    label_dim = latent_dim

    # Convert to tensor

    """ The conversion of signals into tensors is necessary to be able to use the data within TensorFlow """

    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    xy_train = tf.convert_to_tensor(xy_train, dtype=tf.float32)
    xy_val = tf.convert_to_tensor(xy_val, dtype=tf.float32)
    xy_test = tf.convert_to_tensor(xy_test, dtype=tf.float32)

    # Dimention
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    val_size = x_val.shape[0]

    # Data Set

    """ The from_tensor_slices() method takes in an input tensor (xy_train, xy_val, or xy_test) 
        and creates a dataset where each element of the tensor becomes an element of the dataset """

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(xy_train)
        .shuffle(train_size)
        .batch(batch_size, drop_remainder=True)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(xy_val).batch(
        batch_size, drop_remainder=True
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(xy_val).batch(
        batch_size, drop_remainder=True
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        latent_dim,
        label_dim,
        input_dim,
        train_labels,
    )


"""
conv_architectures (list): A list of convolutional layer configurations for the encoder network. Each element in the list
    should be a list of dictionaries, where each dictionary represents the keyword arguments for a `tf.keras.layers.Conv1D`
    layer. The convolutional layers are applied in the order they appear in the list.
"""

conv_architectures = [

        {
            "filters": 64,
            "kernel_size": 3,
            "strides": 2,
            "activation": "tanh",
            "padding": "valid",
        },
        {
            "filters": 128,
            "kernel_size": 3,
            "strides": 2,
            "activation": "tanh",
            "padding": "valid",
        },
        {
            "filters": 256,
            "kernel_size": 3,
            "strides": 2,
            "activation": "tanh",
            "padding": "valid",
        },
    ],

]

"""
linear_architectures (list): A list of dense layer configurations for the decoder network. Each element in the list should
    be a list of dictionaries, where each dictionary represents the keyword arguments for a `tf.keras.layers.Dense` layer.
    The dense layers are applied in the order they appear in the list.
"""
linear_architectures = [
    [
        {"units": 256, "activation": "relu"},
        {"units": 128, "activation": "relu"},
    ],

]




"""  LOSS FUNCTION  """"

def log_normal_pdf(sample, mean, logvar, raxis=1):

    """
        Compute the log probability density function of a normal distribution.

        Args:
            sample (tf.Tensor): Sampled values.
            mean (tf.Tensor): Mean of the distribution.
            logvar (tf.Tensor): Log variance of the distribution.
            raxis (int): Axis to reduce.

        Returns:
            tf.Tensor: Log probability density function.
    """    
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x, input_dim):

    """
        Computes the loss function for the CVAE model.

        Args:
            model (CVAE): CVAE model.
            x (tf.Tensor): Input tensor.
            input_dim (int): Dimensionality of the input signal.

        Returns:
            tf.Tensor: Loss value.
    """    
    x_x = x[:, :input_dim, :] # corresponds to the PPG signal part of the input.
    y = x[:, input_dim:, 0] # corresponds to the part relating to the labels.
    x = x_x
    x_logit, z, mean, logvar = model(x, y) 
    cross_ent = (x_logit - tf.squeeze(x, -1)) ** 2  # mse(x_logit, x) #MSE between input and output.
    
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1]) # Represents the log conditional probability density of the PPG signal.
    logpz = log_normal_pdf(z, 0.0, 0.0)  # Represents the log of the probability density of the latent vector. Normal Distribution. 
    logqz_x = log_normal_pdf(z, mean, logvar) # Represents the log of the probability density of the latent vector distribution.
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)  


@tf.function
def train_step(model, x, optimizer, input_dim):

   """  
   Performs one training step for the CVAE model.
   Calculate loss, calculate gradients, and apply model weight updates using the optimizer

        Args:
            model (CVAE): CVAE model to be trained.
            x (tf.Tensor): Batch of training data.
            optimizer (tf.keras.optimizers.Optimizer): Optimizer for updating the model weights.
            input_dim (int): Input dimensionality.

        Returns:
               loss
    """    
    with tf.GradientTape() as tape: # To compute gradients of model weights versus loss.
        loss = compute_loss(model, x, input_dim)
    gradients = tape.gradient(loss, model.trainable_variables) # To compute the gradients of the loss with respect to the model weights.
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #Update weights model.
    return loss


def generate_and_save_images(model, epoch, test_sample, input_dim):

    """  Generates and saves the images generated by the model.

        Args:
            model (CVAE): Trained CVAE model.
            epoch (int): Current epoch number.
            test_sample (tf.Tensor): Test data sample.
            input_dim (int): Input dimensionality.

        Returns:
            None
    """

    mean, logvar = model.encode(test_sample[:, :input_dim, :])
    z = model.reparameterize(mean, logvar)
    labels = test_sample[:, input_dim:, 0]
    predictions = model.sample(z, labels)

    _, ax = plt.subplots(test_sample.shape[0], 2, figsize=(12, 8))
    for i in range(predictions.shape[0]):
        ax[i, 0].plot(test_sample[i, :input_dim, 0])
        ax[i, 1].plot(predictions[i, :, 0])

    plt.show()


def train_or_load_model(
    *,
    latent_dim,
    train_dataset,
    test_dataset,
    val_dataset,
    label_dim,
    conv_architectures,
    linear_architectures,
    batch_size,
    input_dim,
    epochs=1,
    num_examples_to_generate=6,
):

   """ 
   Trains or loads a CVAE model.

       Args:
           latent_dim (int): Dimensionality of the latent space.
           train_dataset (tf.data.Dataset): Training dataset.
           test_dataset (tf.data.Dataset): Test dataset.
           val_dataset (tf.data.Dataset): Validation dataset.
           label_dim (int): Dimensionality of the label space.
           conv_architectures (list): List of configurations for the convolutional layers in the encoder/decoder network.
           linear_architectures (list): List of configurations for the linear layers in the encoder/decoder network.
           batch_size (int): Batch size.
           input_dim (int): Dimensionality of the input.
           epochs (int, optional): Number of training epochs. Default is 1.
           num_examples_to_generate (int, optional): Number of examples to generate during training. Default is 6.

       Returns:
           CVAE: Trained CVAE model.
    """   
    
    train_log_dir = "logs/"
    model = CVAE(
        latent_dim, label_dim, conv_architectures, linear_architectures, input_dim
    )
    num_examples_to_generate = 6

    if not os.path.exists("trained_model.index"):
        writer = tf.summary.create_file_writer(train_log_dir)
        for conv_settings, linear_settings in product(
            conv_architectures, linear_architectures
        ):
            print("---------")
            print(conv_settings)
            print(linear_settings)
            optimizer = tf.keras.optimizers.Adam(1e-4)

            random_vector = tf.random.normal(
                shape=(num_examples_to_generate, latent_dim)
            )

            assert batch_size >= num_examples_to_generate
            for test_batch in test_dataset.take(1):
                test_sample = test_batch[0:num_examples_to_generate, :, :]

            max_patience = 10
            patience = 0
            best_loss = float("inf")

            for epoch in range(1, epochs + 1):
                start_time = time.time()
                train_losses = []
                for train_x in train_dataset:
                    train_loss = train_step(model, train_x, optimizer, input_dim)
                    train_losses.append(train_loss)
                train_losses = np.array(train_losses).mean()
                end_time = time.time()

                val_losses = []
                for val_x in val_dataset:
                    val_losses.append(compute_loss(model, val_x, input_dim))
                val_losses = np.array(val_losses).mean()

                with writer.as_default():
                    tf.summary.scalar("train_loss", train_losses, step=epoch)
                    tf.summary.scalar("val_loss", val_losses, step=epoch)

                if val_losses < best_loss:
                    best_loss = val_losses
                    patience = 0
                    print(f"Saving model")
                    model.save_weights("trained_model")
                else:
                    patience += 1

                display.clear_output(wait=False)
                print(
                    f"Epoch: {epoch}, Val set LOSS: {val_losses}, time elapsed for current epoch: {end_time - start_time}"
                )

    else:
        print(f"Found model, loading it.")
        model.load_weights("trained_model")

    return model


def plot_reconstrcuted_signal(model, test_dataset, input_dim, num_examples_to_generate):

    """
       Plots the reconstructed signals generated by the CVAE model.

       Args:
           model (CVAE): Trained CVAE model.
           test_dataset (tf.data.Dataset): Test dataset.
           input_dim (int): Dimensionality of the input.
           num_examples_to_generate (int): Number of examples to generate.

       Returns:
           None
    """    
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :]

    reconstructed, *_ = model(
        test_sample[:, :input_dim, :], test_sample[:, input_dim:, 0]
    )

    _, ax = plt.subplots(test_sample.shape[0], 2, figsize=(12, 8))
    for i in range(reconstructed.shape[0]):
        ax[i, 0].plot(test_sample[i, :input_dim, 0])
        ax[i, 1].plot(reconstructed[i, :])
    plt.show()


def generate_samples_from_age(model, train_labels, age, n):

   """
       Generates samples from the CVAE model conditioned on a specific age.

       Args:
           model (CVAE): Trained CVAE model.
           train_labels (pd.DataFrame): Training labels used to condition the generation.
           age (int): Target age to condition the generation.
           n (int): Number of samples to generate.

       Returns:
           np.ndarray: Generated samples.
           np.ndarray: Conditioned labels                                            
    """          
    result_x = []
    result_y = []
    idx = random.randint(0, len(train_labels) - 1)
    z = train_labels.iloc[idx, :].copy()
    z["age"] = age
    z = tf.convert_to_tensor(z.to_numpy().reshape(1, -1), dtype=tf.float32)
    predictions = model.sample(labels=z, num_samples=n)
    result_x.append(predictions.numpy().reshape(1, -1))
    result_y.append(z.numpy().reshape(1, -1))
    return np.concatenate(result_x), np.concatenate(result_y)


def save_samples_from_age_range(model, train_labels, min_age, max_age, n):

   """
       Generates and saves samples from a range of ages using the given model.

       Args:
           model (CVAE): Trained CVAE model used for generating samples.
           train_labels (pd.DataFrame): DataFrame of training labels.
           min_age (int): Minimum age value for generating samples (inclusive).
           max_age (int): Maximum age value for generating samples (exclusive).
           n (int): Number of samples to generate for each age.

       Returns:
           None
   """ 
    
    X_generate = []
    Z_generate = []
    for age in range(min_age, max_age):  # ,99):
        x, z = generate_samples_from_age(model, train_labels, age, n)
        X_generate.append(x)
        Z_generate.append(z)
    X_generate = np.concatenate(X_generate)
    Z_generate = np.concatenate(Z_generate)
    with open(f"generated_samples_{min_age}_{max_age}9.pickle", "wb") as f:
        pickle.dump((X_generate, Z_generate), f)
    print("Successfully saved samples")


def main():
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
    ) = get_data(batch_size)
    model = train_or_load_model(
        epochs=1,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_architectures=conv_architectures,
        linear_architectures=linear_architectures,
        batch_size=batch_size,
        input_dim=input_dim,
    )
    save_samples_from_age_range(model, labels, 18, 99, 1000)


if __name__ == "__main__":
    main()
