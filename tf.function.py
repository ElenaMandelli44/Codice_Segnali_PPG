import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from IPython import display
from itertools import product


def train_step(model, x, optimizer, input_dim):
   """  
   Performs one training step for the CVAE model.
   Calculate loss, calculate gradients, and apply model weight updates using the optimizer

        Args:
            model (CVAE): CVAE model to be trained.
            x (tf.Tensor): Batch of training data.
            optimizer (tf.keras.optimizers.Optimizer): Optimizer for updating the model weights.
            input_dim (int): Input dimensionality.

        Returns:
               loss
    """
   with tf.GradientTape() as tape:     # To compute gradients of model weights versus loss
        loss = compute_loss(model, x, input_dim)
   gradients = tape.gradient(loss, model.trainable_variables) # To compute the gradients of the loss with respect to the model weights
   optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #Update weights model
   return loss

def generate_and_save_images(model, epoch, test_sample,input_dim)):
  """
    Generates and saves images using the model during training.

    Args:
    model (CVAE): The neural network model.
    epoch (int): Current epoch number.
    test_sample (ndarray): Test data sample.
    input_dim (int): Input dimensionality.
    
    Returns:
            None
    
    This function takes as input a model, a test sample and the current epoch. Using the model, 
    the function codes the test sample to obtain the means and logarithms of the variances of the posterior distributions. 
    Next, sample points from the latent space and generate predictions using these points. 
    Finally, view the input data and forecast graphs.
  """
  mean, logvar = model.encode(test_sample[:,:input_dim,:])
  z = model.reparameterize(mean, logvar)
  labels = test_sample[:, input_dim:, 0]
  predictions = model.sample(z)
  
  fig, ax = plt.subplots(test_sample.shape[0], 2, figsize=(12, 8))
  for i in range(predictions.shape[0]):
      ax[i,0].plot(test_sample[i, :input_dim, 0])
      ax[i,1].plot(predictions[i, :, 0])
      
    plt.show()
       
      
  def generate_samples(model, sample, n , input_dim):
    """
        Generates samples using the model.

        Args:
        model (CVAE): The neural network model.
        sample (ndarray): Data sample.
        n (int): Number of samples to generate.
        input_dim (int): Dimensionality of the input.
        
        Returns:
        ndarray: Generated samples (predictions).
        ndarray: Corresponding latent vectors for the generated samples.
    """  
    
    
    result_x = []
    result_y = []
    mean, logvar = model.encode(sample[:,:input_dim,:])
    for i in range(n):
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        result_x.append(predictions.numpy())
        result_y.append(z.numpy())
    return np.concatenate(result_x), np.concatenate(result_y)
  
        """ 
         Trains or loads a CVAE model.

             Args:
                 latent_dim (int): Dimensionality of the latent space.
                 train_dataset (tf.data.Dataset): Training dataset.
                 test_dataset (tf.data.Dataset): Test dataset.
                 val_dataset (tf.data.Dataset): Validation dataset.
                 label_dim (int): Dimensionality of the label space.
                 conv_architectures (list): List of configurations for the convolutional layers in the encoder/decoder network.
                 linear_architectures (list): List of configurations for the linear layers in the encoder/decoder network.
                 batch_size (int): Batch size.
                 input_dim (int): Dimensionality of the input.
                 epochs (int, optional): Number of training epochs. Default is 1.
                 num_examples_to_generate (int, optional): Number of examples to generate during training. Default is 6.

             Returns:
                 CVAE: Trained CVAE model.
          """   

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
         writer = tf.summary.create_file_writer(train_log_di
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
                )
    else:
        print(f"Found model, loading it.")
        model.load_weights("trained_model")

    return model



 
