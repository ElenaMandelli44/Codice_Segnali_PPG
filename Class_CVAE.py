import tensorflow as tf
import numpy as np

#Define a Convolutional Variational Autoencoder (CVAE) model using TensorFlow Keras. The model consists of an encoder and decoder network.


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras import backend as K

class CVAE(Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        input_shape = (28, 28, 1) 
        inputs = Input(shape=input_shape)
        x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
        x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = Flatten()(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        return Model(inputs, [z_mean, z_log_var], name='encoder')

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
