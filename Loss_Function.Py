import tensorflow as tf
import numpy as np


def compute_loss(model, x):
    x_x = x[:, :input_dim, :]
    y = x[:, input_dim:, 0]
    x = x_x
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = mse(x_logit, x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    kl_divergence = tf.reduce_mean(logqz_x - logpz)
    
    mse_loss = tf.reduce_mean(cross_ent)
    kl_weight = 0.01  # Weight to balance the importance of KL divergence
    total_loss = mse_loss + kl_weight * kl_divergence
    return mse_loss
