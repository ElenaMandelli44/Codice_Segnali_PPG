
# Synthetic_PPG

The purpose of this project is to generate "synthetic PPG signals" after training a Conditional Variational Autoencoder on a pre-existing database consisting of 327,054 previously acquired PPG signals.
Once the neural network was trained, synthetic signals were generated based on specific information provided, such as the hypothetical age of the patient for whom the signals were being generated.

## Prerequisites
The project is coded in Python and makes use of popular scientific packages like numpy, pandas, matplotlib, sklearn, and more. Therefore, it is highly recommended to install Anaconda for smooth execution.

## Overview on PPG signals

PPG (photoplethysmography) signals are a noninvasive monitoring technique used to measure changes in blood volume in subcutaneous vascular structures. For this purpose, a so-called pulse oximeter is used, which is usually attached to a part of the body where blood vessels are close to the surface, such as the finger or the ear. In our case, the device is attached to the finger.
A light emitting diode (LED) sends light to a part of the body with a strong blood supply, covered by a thin layer of skin, while a photodiode measures the amount of light transmitted or reflected.

## Overview on Project aim

This program aims to train a Convolutional Variational Autoencoder (CVAE) model using TensorFlow Keras. The Convolutional Variational Autoencoder is a type of neural network that can generate new data samples after being trained on an input dataset. The CVAE model consists of a neural network architecture that includes an encoder and a decoder.
The encoder takes an input tensor, applies a series of convolutional layers followed by dense layers, and produces the mean and standard deviation of the latent distribution. The decoder takes a sample from the latent distribution, applies a series of transposed convolutional layers followed by dense layers, and produces a reconstructed output tensor.
The program loads the training, validation, and test data from pickle files and converts them into tensors used for model training. The model is trained using the training dataset and evaluated using the validation dataset. The Adam optimizer is used to update the model weights during training
During training, the program displays the current epoch, the loss on the validation dataset, and the time elapsed for the current epoch. An early stopping criterion based on the validation dataset loss is also used to stop training if the loss does not improve for a certain number of consecutive epochs.



