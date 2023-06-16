

import numpy as np
import tensorflow as tf
import pickle
import pandas as pd


# Definition of working directory
working_dir = "/Users/elenamandelli/Desktop/PPG/"

# Loading training data
with open(working_dir + "train_db_1p.pickle", "rb") as file:
    df = pickle.load(file)
    
train_labels = pd.DataFrame(df["labels"])
train = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
train = np.expand_dims(train, axis=-1)

# Loading validation data 
with open(working_dir + "validation_db_1p.pickle", "rb") as file:
    df = pickle.load(file)
    
validation_labels = pd.DataFrame(df["labels"])
validation = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
validation = np.expand_dims(validation, axis=-1)

# Loading test data
with open(working_dir + "test_db_1p.pickle", "rb") as file:
    df = pickle.load(file)
    
test_labels = pd.DataFrame(df["labels"])
test = np.asarray([d/np.max(np.abs(d)) for d in df["samples"]])
test = np.expand_dims(test, axis=-1)

# Definition of data for training, validation, and testing.
x_train = train[::100]
x_test = test[::100]
x_val = validation[::100]
y_train = train_labels[::100]
y_test = test_labels [::100]
y_val = validation_labels[::100]


# Expand dimention labels

""" Expanding the dimensions is mandatory to be able to combine the labels 
        with the features during the training phase """

y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

# Combine
xy_train = np.hstack([x_train,y_train])
xy_val = np.hstack([x_val,y_val])
xy_test = np.hstack([x_test,y_test])

input_dim = x_train.shape[1]
latent_dim = y_train.shape[1]
label_dim = latent_dim

# Convert to tensor

""" The conversion of signals into tensors is necessary to be able to use the data within TensorFlow """

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
xy_train = tf.convert_to_tensor(xy_train, dtype=tf.float32)
xy_val = tf.convert_to_tensor(xy_val, dtype=tf.float32)
xy_test = tf.convert_to_tensor(xy_test, dtype=tf.float32)

# Dimention
train_size = x_train.shape[0]
batch_size = 64
test_size = x_test.shape[0]
val_size = x_val.shape[0]

# Data Set

""" The from_tensor_slices() method takes in an input tensor (xy_train, xy_val, or xy_test) 
    and creates a dataset where each element of the tensor becomes an element of the dataset """

train_dataset = (tf.data.Dataset.from_tensor_slices(xy_train).shuffle(train_size).batch(batch_size, drop_remainder=True))
val_dataset = (tf.data.Dataset.from_tensor_slices(xy_val).batch(batch_size, drop_remainder=True))
test_dataset = (tf.data.Dataset.from_tensor_slices(xy_val).batch(batch_size, drop_remainder=True))
