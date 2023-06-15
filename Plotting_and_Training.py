import matplotlib.pyplot as plt


def generate_and_save_images(model, epoch, test_sample):
  """
    Generates and saves images using the model during training.

    Args:
    model (CVAE): The neural network model.
    epoch (int): Current epoch number.
    test_sample (ndarray): Test data sample.
  """
  mean, logvar = model.encode(test_sample[:,:input_dim,:])
  #print(mean)
  #print(test_sample[:,input_dim:,0])
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig, ax = plt.subplots(test_sample.shape[0], 2, figsize=(12, 8))

  for i in range(predictions.shape[0]):
    ax[i,0].plot(test_sample[i, :input_dim, 0])
    ax[i,1].plot(predictions[i, :, 0])

  plt.show()

  
  def generate_samples(model, sample, n):
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

 

# set the dimensionality of the latent space to a plane for visualization later
num_examples_to_generate = 4

 


for conv_settings, linear_settings in product(conv_architectures, linear_architectures):
    print('---------')
    print(conv_settings)
    print(linear_settings)
    optimizer = tf.keras.optimizers.Adam(1e-4)


    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.


    model = CVAE(latent_dim, conv_architectures[0], linear_architectures[0])


 

  
    # Pick a sample of the test set for generating output images

    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
      test_sample = test_batch[0:num_examples_to_generate, :, :]


    #generate_and_save_images(model, 0, test_sample)
    max_patience = 3
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
