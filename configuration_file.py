import tensorflow as tf

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

