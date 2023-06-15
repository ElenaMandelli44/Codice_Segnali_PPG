import matplotlib.pyplot as plt


def generate_and_save_images(model, epoch, test_sample):
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
