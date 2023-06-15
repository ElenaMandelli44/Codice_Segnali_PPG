import tensorflow as tf
import numpy as np

#Define a Convolutional Variational Autoencoder (CVAE) model using TensorFlow Keras. The model consists of an encoder and decoder network.


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras import backend as K

class CVAE(Model):
    
    """"
        The encoder network takes an input tensor, applies a series of convolutional layers followed by dense layers,
        and outputs the mean and log variance of the latent distribution.

        The decoder network takes a sample from the latent distribution, applies a series of transposed convolutional layers
        followed by dense layers, and outputs a reconstructed input tensor.
    
    """"
   
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

# The CVAE model can also be used to generate new samples from the learned distribution using the sample() method.


  @tf.function
  def sample(self, eps=None):
               
       """Generate samples from the learned distribution.

        Args:
            eps (tf.Tensor): Random samples from the latent distribution. If None, new samples are generated.

        Returns:
            tf.Tensor: Generated samples.
        """         
               
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
      """
        Encode the input data and compute mean and log variance of the latent distribution.

        Args:
            x (tf.Tensor): Input data.

        Returns:
            tuple: Mean and log variance of the latent distribution.
        """         
               
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
      """
        Reparameterization trick for sampling from the latent distribution.

        Args:
            mean (tf.Tensor): Mean of the latent distribution.
            logvar (tf.Tensor): Log variance of the latent distribution.

        Returns:
            tf.Tensor: Sampled latent vector.
        """
               
      
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
       """
        Decode the latent vector and reconstruct the input data.

        Args:
            z (tf.Tensor): Latent vector.
            apply_sigmoid (bool): Whether to apply sigmoid activation to the output.

        Returns:
            tf.Tensor: Reconstructed input data.
        """        
               
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
