

def test_make_mean_and_logvar():
    """
    Test the make_mean_and_logvar method of the CVAE class.

    This function creates an instance of the CVAE class and tests its make_mean_and_logvar method.
    It checks if the mean and log variance tensors have the expected shape and type.

    Returns:
        None
    """

    # Define the input parameters for CVAE initialization
    latent_dim = 10
    label_dim = 10
    conv_architectures = [
        [
            {"filters": 64, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"activation": "relu"},
        ],
        [
            {"filters": 128, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"activation": "relu"},
        ],
        [
            {"filters": 256, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"activation": "relu"},
        ],
    ]
    linear_architectures = [
        [{"units": 256, "activation": "relu"}],
        [{"units": 128, "activation": "relu"}],
    ]
    input_dim = 1024

    # Create an instance of CVAE
    cvae = CVAE(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_layers_settings=conv_architectures,
        linear_layers_settings=linear_architectures,
        input_dim=input_dim,
    )

    # Set the random seed
    tf.random.set_seed(42)
    np.random.seed(42)

  
    # Define the input tensor
    x = tf.random.normal(shape=(64, input_dim))

    # Call the make_mean_and_logvar method
    mean, logvar = cvae.make_mean_and_logvar(x)

    # Check the shape of the mean and logvar tensors
    assert mean.shape == (64, latent_dim)
    assert logvar.shape == (64, latent_dim)

    # Check if the mean and logvar tensors are of type tf.Tensor
    assert isinstance(mean, tf.Tensor)
    assert isinstance(logvar, tf.Tensor)


# Run the test function
test_make_mean_and_logvar()


  
