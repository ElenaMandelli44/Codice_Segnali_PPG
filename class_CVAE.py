
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Reshape, Conv1DTranspose
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
   
    def __init__(self,*, latent_dim, label_dim, conv_layers_settings, linear_layers_settings, input_dim):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.label_dim = label_dim
            self.input_dim = input_dim
            self.encoder = self.build_encoder(conv_layers_settings, linear_layers_settings)
            self.decoder = self.build_decoder(conv_layers_settings, linear_layers_settings)
            
    def build_encoder(self, conv_architectures, linear_architectures):
            conv_layers = []

            conv_settings = conv_architectures[0]  

            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[0]))
            try:
                conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[1]))
            except:
                import ipdb

                ipdb.set_trace()
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[2]))


            linear_layers = []
                    
            linear_settings = linear_architectures [0]
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))

                """
                    A sequential model is created, starting with an input layer (tf.keras.layers.InputLayer) that specifies the shape of the input. 
                    The elements of the conv_layers list are then added to the model.
                    Next, the elements of the linear_layers list are added to the model. 
                    The last layer is a dense layer (tf.keras.layers.Dense) with size 2* latent_dim.
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
            try:
                conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[2]))
            except:
                import ipdb
                ipdb.set_trace()
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[1]))
            conv_layers.append(tf.keras.layers.Conv1D(**conv_settings[0]))


            linear_layers = []

            linear_settings = linear_architectures [0]
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[1]))
            linear_layers.append(tf.keras.layers.Dense(**linear_settings[0]))

                
        """   
            A sequential model is constructed, starting with an input layer (tf.keras.layers.InputLayer) that specifies the input shape. 
            The elements of the linear_layers list are added to the model.
            A Reshape layer is added to reshape the output to match the computed value in the encoder's output.
            Then, the elements of the conv_layers list are added to the model.
            A conv_layer with a single filter is subsequently added.
            Conv1D layer is then added with just one fileter
            A Reshape layer is added to reshape the output to match the input_dim value.
            Another Reshape layer is included to reshape the output to match the dimensions (input_dim, 1).

        """        
            return tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(
                            input_shape=(self.latent_dim + self.label_dim)
                        ),  
                        *linear_layers,
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
            z = self.reparameterize(mean, logvar) #returns the sampled latent variables z.
            x_logit = self.decode(z, y) # #The decode function takes the sampled latent variables z and the conditional information y as input and decodes them to generate the reconstructed output x_logit
            return x_logit, z, mean, logvar
            


        def sample(self, eps=None, labels=None, num_samples=1):
            
            """Generates samples from the CVAE model.

            Generates samples by decoding random or specified latent variables and labels.

            Args:
                eps (tf.Tensor, optional): Latent variables.  A noise tensor sampled from the latent space. 
                                            If not provided, random samples will be generated.
                labels (tf.Tensor, optional): A label tensor. If provided, it is combined with the sampled noise to generate the decoded samples.
                                              If not provided, random labels sampled from a normal distribution are generated.
                num_samples (int, optional): The number of samples to generate. This value is used only if eps and labels are not provided.

            Returns:
                tf.Tensor: Decoded samples.
            """
            
            num_samples = (
                eps.shape[0]
                if eps is not None
                else (labels.shape[0] if labels is not None else num_samples)
            )
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
            return self.decode(eps, labels, apply_sigmoid=True) # apply_sigmoid=True It is useful to apply sigmoid to get a final output that looks like a probability distribution


        def encode(self, x):
            
            """Encodes the input x and returns the mean and log variance of the latent space.

            Args:
                x (tf.Tensor): Input tensor.

            Returns:
                tf.Tensor: Mean of the latent space.
                tf.Tensor: Log variance of the latent space.
            """
            
            x = self.encoder(x)
            x = x[:, 2 * (x.shape[1] // 2)]
            mean, logvar = tf.split(x, num_or_size_splits=2, axis=1) #Division is done to separate this information into two distinct tensors
            return mean, logvar


        def reparameterize(self, mean, logvar):
            
            """Reparameterizes the latent variables using the mean and log variance.

            Args:
                mean (tf.Tensor): Mean of the latent space.
                logvar (tf.Tensor): Log variance of the latent space.

            Returns:
                tf.Tensor: Reparameterized latent variables.
            """
            
            eps = tf.random.normal(shape=tf.shape(mean))
            return (eps * tf.exp(logvar * 0.5) + mean )    #Sampling from the latent distribution



        def decode(self, z, labels, apply_sigmoid=False):
            
            """Decodes the latent variables z and labels into reconstructed outputs.

            Args:
                z (tf.Tensor): Latent variables.
                labels (tf.Tensor): Labels.
                apply_sigmoid (bool, optional): Whether to apply sigmoid activation to the output. Defaults to False.

            Returns:
                tf.Tensor: Reconstructed outputs.
            """

            if z.shape [-1] == 11:
                z = z[:,:-1]
            inputs = tf.concat([z, labels], axis=1)
            x = self.decoder(inputs)
            if apply_sigmoid:
                x = tf.sigmoid(x)
            return x       




