import tensorflow as tf
import numpy as np


def compute_loss(model, x):
    x_x = x[:, :input_dim, :]
    y = x[:, input_dim:, 0]
    x = x_x
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    x_logit = model(x)  # Assuming the model directly outputs the reconstructed input tensor
    
    mse_loss = mse(x_logit, x)
    mse_loss = tf.reduce_mean(mse_loss)
    return mse_loss
