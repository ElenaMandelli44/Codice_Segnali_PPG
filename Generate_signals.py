
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle


# Generate new random signals with the updated model
random_indices = random.sample(range(num_examples_to_generate), num_examples_to_generate)
random_sample = tf.gather(test_sample, random_indices)

generate_and_save_images(model, epoch, random_sample)



def generate_samples_from_age(model, train_labels, age, n):
   result_x = []
   result_y = []
   for i in range(n):
         idx = random.randint(0, len(train_labels)-1)
         z = train_labels.iloc[idx, :].copy()
         z['age'] = age
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
