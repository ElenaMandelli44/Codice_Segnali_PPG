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
   
    def __init__(self, latent_dim, conv_layers_settings, linear_layers_settings):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            conv_layers = []
            for conv_settings in conv_layers_settings:


            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings))

            linear_layers = []
            
                """
                    A sequential model is created, starting with an input layer (tf.keras.layers.InputLayer) that specifies the shape of the input. 
                    The elements of the conv_layers list are then added to the model. 
                    A flattening layer is added to convert the output of the convolutional layers into a vector.

                    Next, the elements of the linear_layers list are added to the model. 
                    The last layer is a dense layer (tf.keras.layers.Dense) with size latent_dim + latent_dim.
                    This layer returns the mean and log variance of the latent distribution.
                 """
            for linear_settings in linear_layers_settings:
                linear_layers.append(tf.keras.layers.Dense(**linear_settings))


        self.encoder = tf.keras.Sequential(
           [
            tf.keras.layers.InputLayer(input_shape=(input_dim, 1)),
            *conv_layers,
            tf.keras.layers.Flatten()
            *linear_layers,
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
         )
                    
           
       encoder_conv_layers_output_shape = self.encoder.layers[len(conv_layers)-1].output_shape[1:]

        """      
        The decoder network takes a sample from the latent distribution, applies a series of transposed 
        convolutional layers followed by dense layers, and outputs a reconstructed input tensor. 
        """
        
        conv_layers = []
        for conv_settings in conv_layers_settings[::-1][1:]:
        conv_layers.append(tf.keras.layers.Conv1DTranspose(**conv_settings))
        
        linear_layers = []
        for linear_settings in linear_layers_settings[::-1]:
        linear_layers.append(tf.keras.layers.Dense(**linear_settings))
       

      self.decoder = tf.keras.Sequential( 
            
            
            
         self.decoder = tf.keras.Sequential(
          [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            *linear_layers,

            tf.keras.layers.Dense(units=encoder_conv_layers_output_shape[0]*encoder_conv_layers_output_shape[1]),
            tf.keras.layers.Reshape(target_shape=encoder_conv_layers_output_shape),
            *conv_layers,
            tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=3, strides=2, padding='valid'),
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

          
          
    
    conv_architectures = [
    
        [{'filters': 32, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
         {'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
         {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'}     
        ],   
    
        [{'filters': 32, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
         {'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
         {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'}     
         ], 
    
        [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
         {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
         {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'}
         ], 
    

    
       [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
        {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'},
        {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'relu', 'padding':'valid'}     
        ],   
    
      [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
       {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'tanh', 'padding':'valid'},
        ], 

      [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'sigmoid', 'padding':'valid'},
       {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'sigmoid', 'padding':'valid'},
       {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'sigmoid', 'padding':'valid'}
        ], 
        
      [{'filters': 64, 'kernel_size':3, 'strides': 2, 'activation':'elu', 'padding':'valid'},
       {'filters': 128, 'kernel_size':3, 'strides': 2, 'activation':'elu', 'padding':'valid'},
       {'filters': 256, 'kernel_size':3, 'strides': 2, 'activation':'elu', 'padding':'valid'}  
        ],   
    ]

    linear_architectures = [
      [{'units':256, 'activation':'relu'},
       {'units':128, 'activation':'relu'},
        ],
    

    
      [{'units':256, 'activation':'relu'},
       {'units':128, 'activation':'relu'},
       {'units':64, 'activation':'relu'},
       ],
    
      [{'units':128, 'activation':'relu'},
       {'units':64, 'activation':'relu'},
       ],
 
     ]
