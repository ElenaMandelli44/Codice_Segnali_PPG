import tensorflow as tf
import numpy as np


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """Compute the log probability density function of a normal distribution.

    Args:
        sample (tf.Tensor): Sampled values.
        mean (tf.Tensor): Mean of the distribution.
        logvar (tf.Tensor): Log variance of the distribution.
        raxis (int): Axis to reduce.

    Returns:
        tf.Tensor: Log probability density function.
        
       
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )

def compute_loss (model, x, input_dim):
    """Compute the loss function for the CVAE model.

        Args:
            model (CVAE): CVAE model.
            x (tf.Tensor): Input data.
            input_dim (int): Dimensionality of the input signal.
        Returns:
            tf.Tensor: Loss value.
            
            
         It calculates the mean squared error (MSE) between the reconstructed input (x_logit) and the original input (x). 
         It also computes the negative log-likelihood of the reconstruction error (logpx_z), 
         the Kullback-Leibler (KL) divergence between the prior and posterior distributions over the latent space (logpz and logqz_x), 
         and the MSE term for an additional output of the encoder (y_loss). The final loss value is the sum of these terms.
         
    """
   
    
    x_x = x[:, :input_dim, :] # corresponds to the PPG signal part of the input
    y = x[:, input_dim:, 0]  # corresponds to the part relating to the labels.
    x = x_x
    x_logit, z, mean, logvar = model(x, y)
    cross_ent = (x_logit - tf.squeeze(x, -1)) ** 2   #MSE between input and output
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1]) #Represents the log conditional probability density of the PPG signal
    logpz = log_normal_pdf(z, 0., 0.)  # Represents the log of the probability density of the latent vector. Normal Distribution 
    logqz_x = log_normal_pdf(z, mean, logvar)  # Represents the log of the probability density of the latent vector distribution
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def train_step(model, x, optimizer):
    """
    Perform one training step.

    Args:
        model (CVAE): CVAE model.
        x (tf.Tensor): Input data.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer.

    Returns:
        tf.Tensor: Loss value.
        
      It uses TensorFlow's GradientTape to record the operations for automatic differentiation. 
      The loss is computed using the compute_loss function, and the gradients of the loss with respect to the trainable variables of the model are calculated.
      The optimizer applies the gradients to update the model's trainable variables.   
        
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
