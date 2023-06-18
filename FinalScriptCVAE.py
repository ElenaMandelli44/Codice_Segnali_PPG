#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:30:48 2023

@author: elenamandelli
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import ipdb

# import tensorflow_probability as tfp
import time
import pickle
import pandas as pd
import random
from IPython import display
from itertools import product
import os
from scipy.stats import spearmanr


def get_data(batch_size):
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

    # test_labels_max = raw_test_labels.max(axis=0)
    # test_labels_min = raw_test_labels.min(axis=0)
    # test_labels = (raw_test_labels - test_labels_min) / (test_labels_max - test_labels_min)
    # def get_standardizer(raw_test_labels):
    #     mean = raw_test_labels.mean(axis=0)
    #     std = raw_test_labels.std(axis=0)
    #
    #     def standardize(x):
    #         return (x - mean) / std
    #
    #     return standardize
    #

    # standardizer = get_standardizer(raw_test_labels)
    # standard_test_labels = standardizer(raw_test_labels)
    test = np.asarray([d / np.max(np.abs(d)) for d in df["samples"]])
    test = np.expand_dims(test, axis=-1)

    # Definition of data for training, validation, and testing.

    """  Due to the large number of files, it was decided to work om a reduced number of signlas """

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
    #    [{'filters': 32, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
    #     {'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
    #     {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'}
    #    ],
    #     [{'filters': 32, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
    #      {'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
    #      {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'}
    #      ],
    [
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
    #   [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
    #     {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
    #     {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'}
    #   ],
    #   [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
    #    {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
    # ],
    #  [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'sigmoid', 'padding':'valid'},
    #    {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'sigmoid', 'padding':'valid'},
    #    {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'sigmoid', 'padding':'valid'}
    #   ],
    #   [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'elu', 'padding':'valid'},
    #    {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'elu', 'padding':'valid'},
    #    {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'elu', 'padding':'valid'}
    #   ],
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
    #  [{'units':256, 'activation':'relu'},
    #   {'units':128, 'activation':'relu'},
    #   {'units':64, 'activation':'relu'},
    # ],
    #  [{'units':128, 'activation':'relu'},
    #  {'units':64, 'activation':'relu'},
    # ]
]


class CVAE(tf.keras.Model):
    def __init__(
        self, latent_dim, label_dim, conv_architectures, linear_architectures, input_dim
    ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.input_dim = input_dim
        self.encoder = self.build_encoder(conv_architectures, linear_architectures)
        self.decoder = self.build_decoder(conv_architectures, linear_architectures)

    def build_encoder(self, conv_architectures, linear_architectures):
        conv_layers = []

        conv_settings = conv_architectures[0]  # Prendi il primo set di layer
        conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[0]))
        conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[1]))
        conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[2]))

        linear_layers = []

        linear_settings = linear_architectures[0]
        linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))
        linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))

        # ipdb.set_trace()
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.input_dim, 1)),
                *conv_layers,
                *linear_layers,
                tf.keras.layers.Dense(2 * self.latent_dim),
            ]
        )

    def build_decoder(self, conv_architectures, linear_architectures):
        conv_layers = []

        conv_settings = conv_architectures[-1]  # Prendi l'ultimo set di layer
        conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings[2]))
        conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings[1]))
        conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings[0]))

        linear_layers = []

        linear_settings = linear_architectures[-1]
        linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))
        linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))

        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=(self.latent_dim + self.label_dim)
                ),  # Aggiungiamo 1 per l'input della label
                *linear_layers,
                # inserts unsqueeze
                tf.keras.layers.Reshape((256, 1)),
                *conv_layers,
                tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=1, strides=1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024),
            ]
        )

    def call(self, x, y):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, y)
        return x_logit, z, mean, logvar

    def sample(self, eps=None, labels=None, num_samples=1):
        num_samples = (
            eps.shape[0]
            if not eps is None
            else (labels.shape[0] if not labels is None else num_samples)
        )
        # if eps is None:
        eps = (
            eps
            if not eps is None
            else tf.random.normal(shape=(num_samples, self.latent_dim))
        )
        labels = (
            labels
            if not labels is None
            else tf.random.normal(shape=(num_samples, self.label_dim))
        )
        return self.decode(eps, labels, apply_sigmoid=True)

    def encode(self, x):
        x = self.encoder(x)
        x = x[:, 2 * (x.shape[1] // 2)]
        # [:, :126]
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, labels, apply_sigmoid=False):
        inputs = tf.concat([z, labels], axis=1)
        x = self.decoder(inputs)
        if apply_sigmoid:
            x = tf.sigmoid(x)
        return x


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x, input_dim):
    x_x = x[:, :input_dim, :]
    y = x[:, input_dim:, 0]
    x = x_x
    # x_logit = model.decode(z, y)
    x_logit, z, mean, logvar = model(x, y)

    cross_ent = (x_logit - tf.squeeze(x, -1)) ** 2  # mse(x_logit, x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
    # y_loss = mse(y, mean)
    # y_loss = tf.reduce_mean(y_loss)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)  # + y_loss


@tf.function
def train_step(model, x, optimizer, input_dim):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, input_dim)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def generate_and_save_images(model, epoch, test_sample, input_dim):
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
