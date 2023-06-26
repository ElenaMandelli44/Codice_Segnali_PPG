import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
from IPython import display
from itertools import product
from class_CVAE import CVAE
from configuration_file import conv_architectures,linear_architectures 


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """Compute the log probability density function of a normal distribution.

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
    """Compute the loss function for the CVAE model.

    Args:
        model (CVAE): CVAE model.
        x (tf.Tensor): Input data.
        input_dim (int): Dimensionality of the input signal.
    Returns:
        tf.Tensor: Loss value.


     It calculates the mean squared error (MSE) between the reconstructed input (x_logit) and the original input (x).
     It also computes the negative log-likelihood of the reconstruction error (logpx_z),
     the Kullback-Leibler (KL) divergence between the prior and posterior distributions over the latent space (logpz and logqz_x),
     and the MSE term for an additional output of the encoder (y_loss). The final loss value is the sum of these terms.

    """

    x_x = x[:, :input_dim, :]  # corresponds to the PPG signal part of the input
    y = x[:, input_dim:, 0]  # corresponds to the part relating to the labels.
    x = x_x
    x_logit, z, mean, logvar = model(x, y)
    cross_ent = (x_logit - tf.squeeze(x, -1)) ** 2  # MSE between input and output
    logpx_z = -tf.reduce_sum(
        cross_ent, axis=[1]
    )  # Represents the log conditional probability density of the PPG signal
    logpz = log_normal_pdf(
        z, 0.0, 0.0
    )  # Represents the log of the probability density of the latent vector. Normal Distribution
    logqz_x = log_normal_pdf(
        z, mean, logvar
    )  # Represents the log of the probability density of the latent vector distribution
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


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
    with tf.GradientTape() as tape:  # To compute gradients of model weights versus loss
        loss = compute_loss(model, x, input_dim)
    gradients = tape.gradient(
        loss, model.trainable_variables
    )  # To compute the gradients of the loss with respect to the model weights
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )  # Update weights model
    return loss



def generate_images(model, test_sample, input_dim):
    """
    Generates images using the model.

    Args:
    model (CVAE): The neural network model.
    test_sample (ndarray): Test data sample.
    input_dim (int): Input dimensionality.

    Returns:
    predictions (ndarray): Generated predictions.

    This function takes as input a model and a test sample. Using the model,
    the function codes the test sample to obtain the means and logarithms of the variances of the posterior distributions.
    Next, sample points from the latent space and generate predictions using these points.
    Finally, it returns the generated predictions.
    """
    mean, logvar = model.encode(test_sample[:, :input_dim, :])
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    return predictions


def save_images(predictions, directory):
    """
    Saves the generated images to a directory.

    Args:
    predictions (ndarray): Generated predictions.
    directory (str): Directory path to save the images.

    Returns:
    None

    This function takes as input the generated predictions and a directory path.
    It saves each generated image in the predictions array to the specified directory.
    """
    os.makedirs(directory, exist_ok=True)

    for i in range(predictions.shape[0]):
        plt.plot(predictions[i, :, 0])
        filename = os.path.join(directory, f"image_{i}.png")
        plt.savefig(filename)
        plt.close()



def generate_samples(model, sample, n, input_dim):
    """
    Generates samples using the model.

    Args:
    model (CVAE): The neural network model.
    sample (ndarray): Data sample.
    n (int): Number of samples to generate.
    input_dim (int): Dimensionality of the input.

    Returns:
    ndarray: Generated samples (predictions).
    ndarray: Corresponding latent vectors for the generated samples.
    """

    result_x = []
    result_y = []
    mean, logvar = model.encode(sample[:, :input_dim, :])
    for i in range(n):
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        result_x.append(predictions.numpy())
        result_y.append(z.numpy())
    return np.concatenate(result_x), np.concatenate(result_y)



def train_model(
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
    Trains a CVAE model.

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
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    writer = tf.summary.create_file_writer(train_log_dir)
    for conv_settings, linear_settings in product(
        conv_architectures, linear_architectures
    ):
        print("---------")
        print(conv_settings)
        print(linear_settings)
        optimizer = tf.keras.optimizers.Adam(1e-4)

        random_vector = tf.random.normal(shape=(num_examples_to_generate, latent_dim))

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

    return model


def load_model(latent_dim, label_dim, conv_architectures, linear_architectures, input_dim):
    """
    Loads a pre-trained CVAE model.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        label_dim (int): Dimensionality of the label space.
        conv_architectures (list): List of configurations for the convolutional layers in the encoder/decoder network.
        linear_architectures (list): List of configurations for the linear layers in the encoder/decoder network.
        input_dim (int): Dimensionality of the input.

    Returns:
        CVAE: Loaded CVAE model.
    """
    model = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

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

 
