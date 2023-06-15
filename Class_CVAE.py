import tensorflow as tf
import numpy as np

#Define a Convolutional Variational Autoencoder (CVAE) model using TensorFlow Keras. The model consists of an encoder and decoder network.


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras import backend as K

class CVAE(Model):
    
    #Define a Convolutional Variational Autoencoder (CVAE) model using TensorFlow Keras. The model consists of an encoder and decoder network.
    
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
                [
            tf.keras.layers.InputLayer(input_shape=(input_dim, 1)),
            tf.keras.layers.Conv1D(
                filters=8, kernel_size=2, strides=2, padding='valid', activation='relu'),
            tf.keras.layers.Conv1D(
                filters=16, kernel_size=2, strides=2, padding='valid', activation='relu'),
            tf.keras.layers.Conv1D(
            
            # No activation
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
         )
           
 # The decoder network takes a sample from the latent distribution, applies a series of transposed convolutional layers 
  # followed by dense layers, and outputs a reconstructed input tensor.          
           
           
         self.decoder = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
              tf.keras.layers.Dense(16, activation='relu'),
              tf.keras.layers.Dense(8, activation='relu'),
              tf.keras.layers.Dense(units=12*32, activation='relu'),
              tf.keras.layers.Reshape(target_shape=(12, 32)),
              tf.keras.layers.Conv1DTranspose(
                  filters=8, kernel_size=3, strides=2, padding='valid',
                  activation='relu'),
              tf.keras.layers.Conv1DTranspose(
                  filters=16, kernel_size=3, strides=2, padding='valid',
                  activation='relu'),
          
              # tf.keras.layers.Reshape(target_shape=(batch_size, -1, 1)),
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
         
              tf.keras.layers.Resizing(input_dim, 1),
              tf.keras.layers.Reshape(target_shape=(input_dim, 1)),
          ]
      )              

    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(7*7*64, activation='relu')(latent_inputs)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        outputs = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        return Model(latent_inputs, outputs, name='decoder')

    def sample(self, z_mean, z_log_var):
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sample(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed
