
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


def save_samples_from_age_range(model, train_labels, min_age, max_age, n):
    """
       Generates and saves samples from a range of ages using the given model.

       Args:
           model (CVAE): Trained CVAE model used for generating samples.
           train_labels (pd.DataFrame): DataFrame of training labels.
           min_age (int): Minimum age value for generating samples (inclusive).
           max_age (int): Maximum age value for generating samples (exclusive).
           n (int): Number of samples to generate for each age.

       Returns:
           None
    """   
  
    X_generate = []
    Z_generate = []
    for age in range(min_age, max_age):
        x, z = generate_samples_from_age(model, train_labels, age, n)
        X_generate.append(x)
        Z_generate.append(z)
    X_generate = np.concatenate(X_generate)
    Z_generate = np.concatenate(Z_generate)
    with open(f"generated_samples_{min_age}_{max_age}2.pickle", "wb") as f:
        pickle.dump((X_generate, Z_generate), f)
    print("Successfully saved samples")

