
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras import backend as K



    """
    conv_architectures (list): A list of convolutional layer configurations for the encoder network. Each element in the list
        should be a list of dictionaries, where each dictionary represents the keyword arguments for a `tf.keras.layers.Conv1D`
        layer. The convolutional layers are applied in the order they appear in the list.
    """      
    
    conv_architectures = [

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
            
        ]
   
     ]

    """
    linear_architectures (list): A list of dense layer configurations for the decoder network. Each element in the list should
        be a list of dictionaries, where each dictionary represents the keyword arguments for a `tf.keras.layers.Dense` layer.
        The dense layers are applied in the order they appear in the list.
    """
    
    linear_architectures = [
      [
          {'units':256, 'activation':'relu'},
          {'units':128, 'activation':'relu'},
        ],

     ]




class CVAE(Model):
    
    """"
    This class implements a convolutional variational autoencoder (CVAE) model. The encoder network takes an input tensor,
    applies a series of convolutional layers followed by dense layers, and outputs the mean and log variance of the latent
    distribution. The decoder network then takes a sampled latent variable and reconstructs the original input.

    Args:
        latent_dim (int): The dimensionality of the latent space.
        label_dim (int): The dimensionality of the label space.
        conv_architectures (list): A list of convolutional layer configurations for the encoder network. Each element
                                    in the list should be a tuple containing the keyword arguments for `tf.keras.layers.Conv1D`.
        linear_architectures (list): A list of dense layer configurations for the decoder network. Each element in the
                                    list should be a tuple containing the keyword arguments for `tf.keras.layers.Dense`.
        input_dim (int) : The dimensionality of the input signal.
         
         
    Attributes:
        latent_dim (int): The dimensionality of the latent space.
        label_dim (int): The dimensionality of the label space.
        conv_layers (list): A list of convolutional layers in the encoder network.
        linear_layers (list): A list of dense layers in the decoder network.
        input_dim (int) :The dimensionality of the input signal.
        encoder (tf.keras.Sequential): The encoder network of the CVAE model.
        decoder (tf.keras.Sequential): The decoder network of the CVAE model.

    Methods:
        call(x, y): Executes the forward pass of the CVAE model given an input x and label y, returning the reconstructed
            output, latent variables, mean, and log variance.
        sample(eps=None, labels=None, num_samples=1): Generates samples from the CVAE model by decoding random or
            specified latent variables and labels.
        encode(x): Encodes the input x and returns the mean and log variance of the latent space.
        reparameterize(mean, logvar): Reparameterizes the latent variables using the mean and log variance.
        decode(z, labels, apply_sigmoid=False): Decodes the latent variables z and labels into reconstructed outputs.

    """"
   
    def __init__(self, latent_dim, label_dim, conv_layers_settings, linear_layers_settings, input_dim):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.label_dim = label_dim
            self.input_dim = input_dim
            self.encoder = self.build_encoder(conv_architectures, linear_architectures)
            self.decoder = self.build_decoder(conv_architectures, linear_architectures)
            
    def build_encoder(self, conv_architectures, linear_architectures):
            conv_layers = []

            conv_settings = conv_architectures[0]  
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[0]))
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[1]))
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[2]))


            linear_layers = []
                    
            linear_settings = linear_architectures [0]
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))

                """
                    A sequential model is created, starting with an input layer (tf.keras.layers.InputLayer) that specifies the shape of the input. 
                    The elements of the conv_layers list are then added to the model. 
                    A flattening layer is added to convert the output of the convolutional layers into a vector.

                    Next, the elements of the linear_layers list are added to the model. 
                    The last layer is a dense layer (tf.keras.layers.Dense) with size latent_dim + latent_dim.
                    This layer returns the mean and log variance of the latent distribution.
                 """
             return tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(input_shape=(self.input_dim, 1)),
                            *conv_layers,
                            *linear_layers,
                            tf.keras.layers.Dense(2 * self.latent_dim),
                        ]
                    )

           
       encoder_conv_layers_output_shape = self.encoder.layers[len(conv_layers)-1].output_shape[1:]

        """      
        Decoder network for the convolutional variational autoencoder (CVAE).

        This network takes a sample from the latent distribution, applies a series of transposed convolutional layers
        followed by dense layers, and outputs a reconstructed input tensor.

        Args:
            conv_architectures (list): A list of transposed convolutional layer configurations for the decoder network.
                Each element in the list should be a tuple containing the keyword arguments for `tf.keras.layers.Conv1D`.
            linear_architectures (list): A list of dense layer configurations for the decoder network. Each element in
                the list should be a tuple containing the keyword arguments for `tf.keras.layers.Dense`.

        Attributes:
            conv_layers (list): A list of transposed convolutional layers in the decoder network.
            linear_layers (list): A list of dense layers in the decoder network. 
        """
        
        
        
        def build_decoder(self, conv_architectures, linear_architectures):
            conv_layers = []   
                    

            conv_settings = conv_architectures[0]  
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[2]))
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[1]))
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[0]))


            linear_layers = []

            linear_settings = linear_architectures [0]
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))

                
        """   
            A sequential model is constructed, starting with an input layer (tf.keras.layers.InputLayer) that specifies the input shape. 
            The elements of the linear_layers list are added to the model.
            A dense layer is added with the dimension of the final layer in the encoder.
            A Reshape layer is added to reshape the output to match the computed value in the encoder's output.
            Then, the elements of the conv_layers list are added to the model.
            A conv_layer with a single filter is subsequently added.
            A Lambda layer is included to expand the dimensions of the output by adding an axis to the last dimension.
            A Reshape layer is added to reshape the output to match the input_dim value.
            Another Reshape layer is included to reshape the output to match the dimensions (input_dim, 1).

        """        
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
            

# The CVAE model can also be used to generate new samples from the learned distribution using the sample() method.
 


        def call(self, x, y):
        
            """Forward pass of the CVAE model.

            Encodes the input x, reparameterizes the latent variables, decodes them with the provided label y,
            and returns the reconstructed output, latent variables, mean, and log variance.

            Args:
                x (tf.Tensor): Input tensor.
                y (tf.Tensor): Label tensor.

            Returns:
                tf.Tensor: Reconstructed output.
                tf.Tensor: Latent variables.
                tf.Tensor: Mean of the latent distribution.
                tf.Tensor: Log variance of the latent distribution.
            """
            
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z, y)
            return x_logit, z, mean, logvar
            def sample(self, eps=None, labels=None):

       




