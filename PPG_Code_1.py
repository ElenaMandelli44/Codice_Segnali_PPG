#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:18:16 2023

@author: elenamandelli
"""

# import numpy as np
# import tensorflow as tf
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# import time
# from IPython import display
# from itertools import product
# import random
# import os



# def get_data (batch_size): 
#     """
#         This script is designed to load and preprocess data for a PPG (Photoplethysmography) project. 
#         It provides a function called get_data that returns the necessary data and parameters for training a model.

#         Parameters:
#         - batch_size: The desired batch size for training the model.

#         Returns:
#         - train_dataset: TensorFlow dataset containing the training data.
#         - val_dataset: TensorFlow dataset containing the validation data.
#         - test_dataset: TensorFlow dataset containing the test data.
#         - input_dim: Dimension of the input signals.
#         - latent_dim: Dimension of the target labels.
#         - label_dim: Dimension of the target labels.
#         - input_dim: Dimension of the input signals.
#     """

#     # Definition of working directory
#     working_dir = "/Users/elenamandelli/Desktop/PPG_Project/"

#     # Loading training data
#     with open(working_dir + "train_db_1p.pickle", "rb") as file:
#         df = pickle.load(file)  # DataFrame that contains the uploaded data

#     train_labels = pd.DataFrame(df["labels"])
#     train = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]]) #Training signlas are normalized by didviding each signal by the absolute maximum value among all signlas.
#     train = np.expand_dims(train, axis=-1)

#     # Loading validation data 
#     with open(working_dir + "validation_db_1p.pickle", "rb") as file:
#         df = pickle.load(file)

#     validation_labels = pd.DataFrame(df["labels"])
#     validation = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
#     validation = np.expand_dims(validation, axis=-1)

#     # Loading test data
#     with open(working_dir + "test_db_1p.pickle", "rb") as file:
#         df = pickle.load(file)

#     test_labels = pd.DataFrame(df["labels"])
#     test = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
#     test = np.expand_dims(test, axis=-1)

#     # Definition of data for training, validation, and testing.

#     """  Due to the large number of files, it was decided to work om a reduced number of signlas """
#     x_train = train[::100]
#     x_test = test[::100]
#     x_val = validation[::100]
#     y_train = train_labels[::100]
#     y_test = test_labels [::100]
#     y_val = validation_labels[::100]


#     # Expand dimention labels

#     """ Expanding the dimensions is mandatory to be able to combine the labels 
#             with the features during the training phase """

#     y_train = np.expand_dims(y_train, axis=-1)
#     y_test = np.expand_dims(y_test, axis=-1)
#     y_val = np.expand_dims(y_val, axis=-1)

#     # Combine
#     xy_train = np.hstack([x_train,y_train])
#     xy_val = np.hstack([x_val,y_val])
#     xy_test = np.hstack([x_test,y_test])

#     input_dim = x_train.shape[1]
#     latent_dim = y_train.shape[1]
#     label_dim = latent_dim


#     # Convert to tensor

#     """ The conversion of signals into tensors is necessary to be able to use the data within TensorFlow """

#     x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
#     y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
#     x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
#     y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
#     x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
#     y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
#     xy_train = tf.convert_to_tensor(xy_train, dtype=tf.float32)
#     xy_val = tf.convert_to_tensor(xy_val, dtype=tf.float32)
#     xy_test = tf.convert_to_tensor(xy_test, dtype=tf.float32)

#     # Dimention
#     train_size = x_train.shape[0]
#     test_size = x_test.shape[0]
#     val_size = x_val.shape[0]


#     # Data Set

#     """ The from_tensor_slices() method takes in an input tensor (xy_train, xy_val, or xy_test) 
#         and creates a dataset where each element of the tensor becomes an element of the dataset """

#     train_dataset = (tf.data.Dataset.from_tensor_slices(xy_train).shuffle(train_size).batch(batch_size, drop_remainder=True))
#     val_dataset = (tf.data.Dataset.from_tensor_slices(xy_val).batch(batch_size, drop_remainder=True))
#     test_dataset = (tf.data.Dataset.from_tensor_slices(xy_val).batch(batch_size, drop_remainder=True))

#     return (
#             train_dataset,
#             val_dataset,
#             test_dataset,
#             input_dim,
#             latent_dim,
#             label_dim,
#             input_dim,
#             train_labels,
#         )


# """
#     conv_architectures (list): A list of convolutional layer configurations for the encoder network. Each element in the list
#     should be a list of dictionaries, where each dictionary represents the keyword arguments for a `tf.keras.layers.Conv1D`
#     layer. The convolutional layers are applied in the order they appear in the list.
# """

# conv_architectures = [

# [
#     {
#         "filters": 64,
#         "kernel_size": 3,
#         "strides": 2,
#         "activation": "tanh",
#         "padding": "valid",
#     },
#     {
#         "filters": 128,
#         "kernel_size": 3,
#         "strides": 2,
#         "activation": "tanh",
#         "padding": "valid",
#     },
#     {
#         "filters": 256,
#         "kernel_size": 3,
#         "strides": 2,
#         "activation": "tanh",
#         "padding": "valid",
#     },
# ], 

# ]

# """
#     linear_architectures (list): A list of dense layer configurations for the decoder network. Each element in the list should
#     be a list of dictionaries, where each dictionary represents the keyword arguments for a `tf.keras.layers.Dense` layer.
#     The dense layers are applied in the order they appear in the list.
# """
# linear_architectures = [
# [
#     {'units':256, 'activation':'relu'},
#     {'units':128, 'activation':'relu'},
# ],
    
# ]

# class CVAE(tf.keras.Model):
#    """
#     Convolutional Variational Autoencoder (CVAE) model.

#     This class implements a Convolutional Variational Autoencoder (CVAE) model, which consists of an encoder network,
#     a decoder network, and methods for encoding, decoding, and sampling from the latent space.

#     Args:
#         latent_dim (int): The dimensionality of the latent space.
#         label_dim (int): The dimensionality of the label space.
#         conv_architectures (list): A list of convolutional layer configurations for the encoder and decoder networks.
#             Each element in the list should be a tuple containing the keyword arguments for `tf.keras.layers.Conv1D` or
#             `tf.keras.layers.Conv1DTranspose`, depending on the network.
#         linear_architectures (list): A list of dense layer configurations for the encoder and decoder networks.
#             Each element in the list should be a tuple containing the keyword arguments for `tf.keras.layers.Dense`.
#         input_dim (int): The dimensionality of the input signal.

#     Attributes:
#         latent_dim (int): The dimensionality of the latent space.
#         label_dim (int): The dimensionality of the label space.
#         input_dim (int): The dimensionality of the input signal.
#         encoder (tf.keras.Sequential): The encoder network of the CVAE model.
#         decoder (tf.keras.Sequential): The decoder network of the CVAE model.

#     Methods:
#         call(x, y): Executes the forward pass of the CVAE model given an input x and label y, returning the reconstructed
#             output, latent variables, mean, and log variance.
#         sample(eps=None, labels=None, num_samples=1): Generates samples from the CVAE model by decoding random or
#             specified latent variables and labels.
#         encode(x): Encodes the input x and returns the mean and log variance of the latent space.
#         reparameterize(mean, logvar): Reparameterizes the latent variables using the mean and log variance.
#         decode(z, labels, apply_sigmoid=False): Decodes the latent variables z and labels into reconstructed outputs.

#     """

#    def __init__(self, latent_dim, label_dim, conv_architectures , linear_architectures, input_dim):
#         super(CVAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.label_dim = label_dim
#         self.input_dim = input_dim
#         self.encoder = self.build_encoder(conv_architectures, linear_architectures)
#         self.decoder = self.build_decoder(conv_architectures, linear_architectures)
        
#    def build_encoder(self, conv_architectures, linear_architectures):
#         conv_layers = []

#         conv_settings = conv_architectures[0]  
#         conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[0]))
#         conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[1]))
#         conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[2]))

#         linear_layers = []

#         linear_settings = linear_architectures[0]
#         linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))
#         linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))
        
#         """
#             A sequential model is created, starting with an input layer (tf.keras.layers.InputLayer) that specifies the shape of the input. 
#             The elements of the conv_layers list are then added to the model. 
           
#             Next, the elements of the linear_layers list are added to the model. 
#             The last layer is a dense layer (tf.keras.layers.Dense) with size latent_dim + latent_dim.
#             This layer returns the mean and log variance of the latent distribution.
#         """
        
#         return tf.keras.Sequential(
#             [
#                 tf.keras.layers.InputLayer(input_shape=(self.input_dim, 1)),
#                 *conv_layers,
#                 *linear_layers,
#                 tf.keras.layers.Dense(2 * self.latent_dim),
#             ]
#         )

        
#         """           
#             Decoder network for the convolutional variational autoencoder (CVAE).

#             This network takes a sample from the latent distribution, applies a series of transposed convolutional layers
#             followed by dense layers, and outputs a reconstructed input tensor.

#             Args:
#                 conv_architectures (list): A list of transposed convolutional layer configurations for the decoder network.
#                     Each element in the list should be a tuple containing the keyword arguments for `tf.keras.layers.Conv1D`.
#                 linear_architectures (list): A list of dense layer configurations for the decoder network. Each element in
#                     the list should be a tuple containing the keyword arguments for `tf.keras.layers.Dense`.

#             Attributes:
#                 conv_layers (list): A list of transposed convolutional layers in the decoder network.
#                 linear_layers (list): A list of dense layers in the decoder network.
#         """
        
#         def build_decoder(self, conv_architectures, linear_architectures):
#                 conv_layers = []

#                 conv_settings = conv_architectures[-1] 
#                 conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings[2]))
#                 conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings[1]))
#                 conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings[0]))

#                 linear_layers = []

#                 linear_settings = linear_architectures[-1]
#                 linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))
#                 linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))
                
#         """   
#               A sequential model is constructed, starting with an input layer (tf.keras.layers.InputLayer) that specifies the input shape. 
#               The elements of the linear_layers list are added to the model.
#               A dense layer is added with the dimension of the final layer in the encoder.
#               A Reshape layer is added to reshape the output to match the computed value in the encoder's output.
#               Then, the elements of the conv_layers list are added to the model.
#               Conv1D layer is then added with just one fileter
#               A Flatten layer is added to flatten the output from the convolutional layers into a 1D vector.
#               A Dense layer is added with 1024 units.

#         """
#         return tf.keras.Sequential(
#                 [
#                     tf.keras.layers.InputLayer(
#                         input_shape=(self.latent_dim + self.label_dim)
#                     ),  # Aggiungiamo 1 per l'input della label
#                     *linear_layers,
#                     tf.keras.layers.Reshape((256, 1)),
#                     *conv_layers,
#                     tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=1, strides=1),
#                     tf.keras.layers.Flatten(),
#                     tf.keras.layers.Dense(1024),
#                 ]
#             )   
                
            
            
#         def call(self, x, y):
        
#             """
#                 Forward pass of the CVAE model.

#                 Encodes the input x, reparameterizes the latent variables, decodes them with the provided label y,
#                 and returns the reconstructed output, latent variables, mean, and log variance.

#                 Args:
#                     x (tf.Tensor): Input tensor.
#                     y (tf.Tensor): Label tensor.

#                 Returns:
#                     tf.Tensor: Reconstructed output.
#                     tf.Tensor: Latent variables.
#                     tf.Tensor: Mean of the latent distribution.
#                     tf.Tensor: Log variance of the latent distribution.
#             """
            
#             mean, logvar = self.encode(x)
#             z = self.reparameterize(mean, logvar) #returns the sampled latent variables z.
#             x_logit = self.decode(z, y)   #The decode function takes the sampled latent variables z and the conditional information y as input and decodes them to generate the reconstructed output x_logit
#             return x_logit, z, mean, logvar
       

#         def sample(self, eps=None, labels=None, num_samples=1):
            
#             """
#                 Generates samples from the CVAE model.

#                 Generates samples by decoding random or specified latent variables and labels.

#                 Args:
#                     eps (tf.Tensor, optional): Latent variables. A noise tensor sampled from the latent space.
#                                                 If not provided, random samples will be generated.
#                     labels (tf.Tensor, optional): A label tensor. If provided, it is combined with the sampled noise to generate the decoded samples.
#                                                   If not provided, random labels sampled from a normal distribution are generated.
#                     num_samples (int, optional): The number of samples to generate. This value is used only if eps and labels are not provided.

#                 Returns:
#                     tf.Tensor: Decoded samples.
#             """
            

#             num_samples = (
#                 eps.shape[0]
#                 if not eps is None
#                 else (labels.shape[0] if not labels is None else num_samples)
#             )
#             # if eps is None:
#             eps = (
#                 eps
#                 if not eps is None
#                 else tf.random.normal(shape=(num_samples, self.latent_dim))
#             )
#             labels = (
#                 labels
#                 if not labels is None
#                 else tf.random.normal(shape=(num_samples, self.label_dim))
#             )
#             return self.decode(eps, labels, apply_sigmoid=True)

#         def encode(self, x):
            
#             """
#                 Encodes the input x and returns the mean and log variance of the latent space.

#                 Args:
#                     x (tf.Tensor): Input tensor.

#                 Returns:
#                     tf.Tensor: Mean of the latent space.
#                     tf.Tensor: Log variance of the latent space.
#             """
            
#             x = self.encoder(x)
#             x = x[:, 2 * (x.shape[1] // 2)]
#             mean, logvar = tf.split(x, num_or_size_splits=2, axis=1) #Division is done to separate this information into two distinct tensors
#             return mean, logvar


#         def reparameterize(self, mean, logvar):
            
#             """
#                 Reparameterizes the latent variables using the mean and log variance.

#                 Args:
#                     mean (tf.Tensor): Mean of the latent space.
#                     logvar (tf.Tensor): Log variance of the latent space.

#                 Returns:
#                     tf.Tensor: Reparameterized latent variables.
#             """
            
#             eps = tf.random.normal(shape=tf.shape(mean))
#             return eps * tf.exp(logvar * 0.5) + mean    #Sampling from the latent distribution


#         def decode(self, z, labels, apply_sigmoid=False):
            
#             """
#                 Decodes the latent variables z and labels into reconstructed outputs.

#                 Args:
#                     z (tf.Tensor): Latent variables.
#                     labels (tf.Tensor): Labels.
#                     apply_sigmoid (bool, optional): Whether to apply sigmoid activation to the output. Defaults to False.

#                 Returns:
#                     tf.Tensor: Reconstructed outputs.
#             """
            
#             inputs = tf.concat([z, labels], axis=1)
#             x = self.decoder(inputs)
#             if apply_sigmoid:
#                 x = tf.sigmoid(x)
#             return x



# """ LOSS FUNCTION """
    

# def log_normal_pdf(sample, mean, logvar, raxis=1):
#     """
#         Compute the log probability density function of a normal distribution.

#         Args:
#             sample (tf.Tensor): Sampled values.
#             mean (tf.Tensor): Mean of the distribution.
#             logvar (tf.Tensor): Log variance of the distribution.
#             raxis (int): Axis to reduce.

#         Returns:
#             tf.Tensor: Log probability density function.
#     """
#     log2pi = tf.math.log(2. * np.pi)
#     return tf.reduce_sum(
#         -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#         axis=raxis
#     )

# def compute_loss(model, x, input_dim):
#     """
#         Computes the loss function for the CVAE model.

#         Args:
#             model (CVAE): CVAE model.
#             x (tf.Tensor): Input tensor.
#             input_dim (int): Dimensionality of the input signal.

#         Returns:
#             tf.Tensor: Loss value.
#     """
#     x_x = x[:, :input_dim, :]  # corresponds to the PPG signal part of the input
#     y = x[:, input_dim:, 0] # corresponds to the part relating to the labels.
#     x = x_x
#     x_logit, z, mean, logvar = model(x, y)
#     cross_ent = (x_logit - tf.squeeze(x, -1)) ** 2 #MSE between input and output
    
#     logpx_z = -tf.reduce_sum(cross_ent, axis=[1]) # Represents the log conditional probability density of the PPG signal
#     logpz = log_normal_pdf(z, 0.0, 0.0)  # Represents the log of the probability density of the latent vector. Normal Distribution 
#     logqz_x = log_normal_pdf(z, mean, logvar) # Represents the log of the probability density of the latent vector distribution
#     return -tf.reduce_mean(logpx_z + logpz - logqz_x)




# @tf.function
# def train_step(model, x, optimizer, input_dim):
#    """  
#    Performs one training step for the CVAE model.
#    Calculate loss, calculate gradients, and apply model weight updates using the optimizer

#         Args:
#             model (CVAE): CVAE model to be trained.
#             x (tf.Tensor): Batch of training data.
#             optimizer (tf.keras.optimizers.Optimizer): Optimizer for updating the model weights.
#             input_dim (int): Input dimensionality.

#         Returns:
#                loss
#     """
#    with tf.GradientTape() as tape:     # To compute gradients of model weights versus loss
#         loss = compute_loss(model, x, input_dim)
#    gradients = tape.gradient(loss, model.trainable_variables) # To compute the gradients of the loss with respect to the model weights
#    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #Update weights model
#    return loss
    
# def generate_and_save_images(model, epoch, test_sample, input_dim):
#     """  Generates and saves the images generated by the model.

#         Args:
#             model (CVAE): Trained CVAE model.
#             epoch (int): Current epoch number.
#             test_sample (tf.Tensor): Test data sample.
#             input_dim (int): Input dimensionality.

#         Returns:
#             None
#     """

#     mean, logvar = model.encode(test_sample[:, :input_dim, :])
#     z = model.reparameterize(mean, logvar)
#     labels = test_sample[:, input_dim:, 0]
#     predictions = model.sample(z, labels)

#     fig, ax = plt.subplots(test_sample.shape[0], 2, figsize=(12, 8))
#     for i in range(predictions.shape[0]):
#         ax[i, 0].plot(test_sample[i, :input_dim, 0])
#         ax[i, 1].plot(predictions[i, :, 0])

#     plt.show()
   
    
    
# def train_or_load_model(
#     *,
#     latent_dim,
#     train_dataset,
#     test_dataset,
#     val_dataset,
#     label_dim,
#     conv_architectures,
#     linear_architectures,
#     batch_size,
#     input_dim,
#     epochs=1,
#     num_examples_to_generate=6,
#    ):  
    
#    """ 
#    Trains or loads a CVAE model.

#        Args:
#            latent_dim (int): Dimensionality of the latent space.
#            train_dataset (tf.data.Dataset): Training dataset.
#            test_dataset (tf.data.Dataset): Test dataset.
#            val_dataset (tf.data.Dataset): Validation dataset.
#            label_dim (int): Dimensionality of the label space.
#            conv_architectures (list): List of configurations for the convolutional layers in the encoder/decoder network.
#            linear_architectures (list): List of configurations for the linear layers in the encoder/decoder network.
#            batch_size (int): Batch size.
#            input_dim (int): Dimensionality of the input.
#            epochs (int, optional): Number of training epochs. Default is 1.
#            num_examples_to_generate (int, optional): Number of examples to generate during training. Default is 6.

#        Returns:
#            CVAE: Trained CVAE model.
#     """   

#    train_log_dir = "logs/"
#    model = CVAE(
#         latent_dim, label_dim, conv_architectures, linear_architectures, input_dim
#     )
      
#    num_examples_to_generate = 6

#    if not os.path.exists("trained_model.index"):
#         writer = tf.summary.create_file_writer(train_log_dir)
#         for conv_settings, linear_settings in product(
#             conv_architectures, linear_architectures
#         ):
#             print("---------")
#             print(conv_settings)
#             print(linear_settings)
#             optimizer = tf.keras.optimizers.Adam(1e-4)

#             random_vector = tf.random.normal(
#                 shape=(num_examples_to_generate, latent_dim)
#             )

#             assert batch_size >= num_examples_to_generate
#             for test_batch in test_dataset.take(1):
#                 test_sample = test_batch[0:num_examples_to_generate, :, :]

#             max_patience = 10
#             patience = 0
#             best_loss = float("inf")

#             for epoch in range(1, epochs + 1):
#                 start_time = time.time()
#                 train_losses = []
#                 for train_x in train_dataset:
#                     train_loss = train_step(model, train_x, optimizer, input_dim)
#                     train_losses.append(train_loss)
#                 train_losses = np.array(train_losses).mean()
#                 end_time = time.time()

#                 val_losses = []
#                 for val_x in val_dataset:
#                     val_losses.append(compute_loss(model, val_x, input_dim))
#                 val_losses = np.array(val_losses).mean()

#                 with writer.as_default():
#                     tf.summary.scalar("train_loss", train_losses, step=epoch)
#                     tf.summary.scalar("val_loss", val_losses, step=epoch)

#                 if val_losses < best_loss:
#                     best_loss = val_losses
#                     patience = 0
#                     print(f"Saving model")
#                     model.save_weights("trained_model")
#                 else:
#                     patience += 1

#                 display.clear_output(wait=False)
#                 print(
#                     f"Epoch: {epoch}, Val set LOSS: {val_losses}, time elapsed for current epoch: {end_time - start_time}"
#                 )

#         else:
        
#             print(f"Found model, loading it.")
#             model.load_weights("trained_model")

#         return model



# def plot_reconstrcuted_signal(model, test_dataset, input_dim, num_examples_to_generate):
                                                
#     """
#        Plots the reconstructed signals generated by the CVAE model.

#        Args:
#            model (CVAE): Trained CVAE model.
#            test_dataset (tf.data.Dataset): Test dataset.
#            input_dim (int): Dimensionality of the input.
#            num_examples_to_generate (int): Number of examples to generate.

#        Returns:
#            None
#     """
                                                
#     for test_batch in test_dataset.take(1):
#         test_sample = test_batch[0:num_examples_to_generate, :, :]

#     reconstructed, *_ = model(
#         test_sample[:, :input_dim, :], test_sample[:, input_dim:, 0]
#     )

#     _, ax = plt.subplots(test_sample.shape[0], 2, figsize=(12, 8))
#     for i in range(reconstructed.shape[0]):
#         ax[i, 0].plot(test_sample[i, :input_dim, 0])
#         ax[i, 1].plot(reconstructed[i, :])
#     plt.show()

# def generate_samples_from_age(model, train_labels, age, n):
#    """
#        Generates samples from the CVAE model conditioned on a specific age.

#        Args:
#            model (CVAE): Trained CVAE model.
#            train_labels (pd.DataFrame): Training labels used to condition the generation.
#            age (int): Target age to condition the generation.
#            n (int): Number of samples to generate.

#        Returns:
#            np.ndarray: Generated samples.
#            np.ndarray: Conditioned labels                                            
#     """      
                                                
#    result_x = []
#    result_y = []
#    idx = random.randint(0, len(train_labels) - 1) #This index is used to randomly select a training label from the train_labels DataFrame.
#    z = train_labels.iloc[idx, :].copy()
#    z["age"] = age # Update the age value in the z label with the value provided as input age.
#    z = tf.convert_to_tensor(z.to_numpy().reshape(1, -1), dtype=tf.float32)
#    predictions = model.sample(labels=z, num_samples=n)
#    result_x.append(predictions.numpy().reshape(1, -1))
#    result_y.append(z.numpy().reshape(1, -1))
#    return np.concatenate(result_x), np.concatenate(result_y)

                                                
# def save_samples_from_age_range(model, train_labels, min_age, max_age, n):
#    """
#        Generates and saves samples from a range of ages using the given model.

#        Args:
#            model (CVAE): Trained CVAE model used for generating samples.
#            train_labels (pd.DataFrame): DataFrame of training labels.
#            min_age (int): Minimum age value for generating samples (inclusive).
#            max_age (int): Maximum age value for generating samples (exclusive).
#            n (int): Number of samples to generate for each age.

#        Returns:
#            None
#     """   
         
#    X_generate = []
#    Z_generate = []
#    for age in range(min_age, max_age):
#         x, z = generate_samples_from_age(model, train_labels, age, n)
#         X_generate.append(x)
#         Z_generate.append(z)
#    X_generate = np.concatenate(X_generate)
#    Z_generate = np.concatenate(Z_generate)
#    with open(f"generated_samples_{min_age}_{max_age}2.pickle", "wb") as f:
#         pickle.dump((X_generate, Z_generate), f)
#    print("Successfully saved samples")

   
                                                
                                                
# def main():
#     batch_size = 64
#     (
#         train_dataset,
#         val_dataset,
#         test_dataset,
#         input_dim,
#         latent_dim,
#         label_dim,
#         input_dim,
#         labels,
#     ) = get_data(batch_size)
#     model = train_or_load_model(
#         epochs=300,
#         train_dataset=train_dataset,
#         test_dataset=test_dataset,
#         val_dataset=val_dataset,
#         latent_dim=latent_dim,
#         label_dim=label_dim,
#         conv_architectures=conv_architectures,
#         linear_architectures=linear_architectures,
#         batch_size=batch_size,
#         input_dim=input_dim,
#     )
#     save_samples_from_age_range(model, labels, 18, 99, 1000)


# if __name__ == "__main__":
#     main()
