
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle


def generate_samples_from_age(model, train_labels, age, n):
   
      """
    Generate samples from the model with a specific age.

    Args:
        model (CVAE): The neural network model.
        train_labels (DataFrame): Training labels data.
        age (int): Age value to be used for generating samples.
        n (int): Number of samples to generate.

    Returns:
        ndarray: Generated samples (predictions).
        ndarray: Corresponding latent vectors for the generated samples.
    """
   
   
   result_x = []
   result_y = []
   for i in range(n):
         idx = random.randint(0, len(train_labels)-1) #This index is used to randomly select a training label from the train_labels DataFrame.
         z = train_labels.iloc[idx, :].copy()
         z['age'] = age # Update the age value in the z label with the value provided as input age.
         z = tf.convert_to_tensor(z.to_numpy().reshape(1,-1), dtype=tf.float32)

         predictions = model.sample(z)
         result_x.append(predictions.numpy().reshape(1,-1))
         result_y.append(z.numpy().reshape(1,-1))
   return np.concatenate(result_x), np.concatenate(result_y)


X_generate = []
Z_generate = []
for age in range(18,31):
    x, z = generate_samples_from_age(model, train_labels, age, 688 )
    X_generate.append(x)
    Z_generate.append(z)
X_generate = np.concatenate(X_generate)
Z_generate = np.concatenate(Z_generate)
Z_generate = pd.DataFrame(Z_generate, columns=train_labels.columns)

file_path = 'generate_signal_PPG.pkl'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        existing_data = pickle.load(f)
        
    existing_data['samples'] = np.concatenate([existing_data['samples'], X_generate])
    existing_data['labels'] = pd.concat([existing_data['labels'], Z_generate])

    with open(file_path, 'wb') as f:
        pickle.dump(existing_data, f)
else:
    generate = {'samples': X_generate, 'labels': Z_generate}
    with open(file_path, 'wb') as f:
        pickle.dump(generate, f)

Z_generate
X_generate
