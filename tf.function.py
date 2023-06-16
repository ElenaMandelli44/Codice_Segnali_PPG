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
       
      
  def generate_samples(model, sample, n):
    """
        Generates samples using the model.

        Args:
        model (CVAE): The neural network model.
        sample (ndarray): Data sample.
        n (int): Number of samples to generate.

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
  
  
  
  # PLOTTING

   # epochs 
  epochs = 100


  # set the dimensionality of the latent space to a plane for visualization later
  num_examples_to_generate = 6

  """
  Iterates over the product of convolutional and linear settings and performs the following operations for each combination:
      - Prints a separator line to indicate the current combination.
      - Prints the convolutional settings.
      - Prints the linear settings.

  Parameters:
      conv_architectures (list): A list of convolutional settings.
      linear_architectures (list): A list of linear settings.

  Returns:
      None
  """ 


  for conv_settings, linear_settings in product(conv_architectures, linear_architectures):
      print('---------')
      print(conv_settings)
      print(linear_settings)
      #The Adam optimizer is initialized with a learning rate of 1e-4. ( best value )
      optimizer = tf.keras.optimizers.Adam(1e-4)


      # Generate a new random vector on each iteration
      random_vector = tf.random.normal(shape=(num_examples_to_generate, latent_dim))

      """
      Creates an instance of the CVAE (Conditional Variational Autoencoder) model with the given latent dimension, convolutional settings, and linear settings.

      Parameters:
          latent_dim (int): The dimensionality of the latent space.
          conv_settings (tuple): A tuple containing the convolutional settings for the model.
          linear_settings (tuple): A tuple containing the linear settings for the model.

      Returns:
          model (CVAE): An instance of the CVAE model.
      """

      model = CVAE(latent_dim, conv_settings, linear_settings)




      """
      Selects a sample from the test set to generate output images.

      Parameters:
          batch_size (int): The batch size used for training and testing the model.
          num_examples_to_generate (int): The number of examples to generate output images for.

      Returns:
          test_sample (Tensor): A subset of the test set consisting of num_examples_to_generate examples.
      """

      assert batch_size >= num_examples_to_generate
      for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :]


      #generate_and_save_images(model, 0, test_sample)

      """
      Variables are initialized for early stopping management: 
      max_patience represents the maximum number of consecutive epochs in which the loss value does not improve,
      patience keeps track of consecutive epochs without improvement, and best_loss keeps track of the best loss so far.
      """

      max_patience = 10
      patience = 0 # early stopping
      best_loss = float('inf')

      for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
          train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for val_x in val_dataset:
          loss(compute_loss(model, val_x))
        loss_result = loss.result()
        if loss_result<best_loss:
          best_loss = loss_result
          patience = 0
        else:
          patience +=1
        display.clear_output(wait=False)
        print('Epoch: {}, Val set LOSS: {}, time elapse for current epoch: {}'
              .format(epoch, loss_result, end_time - start_time))
        if patience >= max_patience: break

      loss = tf.keras.metrics.Mean()
      for test_x in test_dataset:
        loss(compute_loss(model, test_x))
      loss_result = loss.result()

      print('Test loss: ', loss_result)
      generate_and_save_images(model, epoch, test_sample)
