
# Synthetic_PPG

The purpose of this project is to generate "synthetic PPG signals" after training a Conditional Variational Autoencoder on a pre-existing database consisting of 327,054 previously acquired PPG signals.
Once the neural network was trained, synthetic signals were generated based on specific information provided, such as the hypothetical age of the patient for whom the signals were being generated.

## Prerequisites
The project is coded in Python and makes use of popular scientific packages like numpy, pandas, matplotlib, sklearn, and more. Therefore, it is highly recommended to install Anaconda for smooth execution.


## Python Installation

First of all ensure that a right Python version is installed (Python >= 3.10.9 is required).
The [Anaconda/Miniconda](https://www.anaconda.com/) python version is recomended.

### Installing prerequisites

To install the prerequisites type:

```
pip install -r ./requirements.txt
```

## Files description

1) main.py -> This file is the main script that executes the program logic.
2) preprocessing.py -> This file contains several useful functions for loading and preparing data.
3) configuration_file.py -> Defines two lists, conv_architectures and linear_architectures, which represent the layer configurations for the encoder and decoder networks, respectively.
4) class_cvae.py ->  Class implementation of a Convolutional Variational Autoencoder (CVAE) in TensorFlow
5) functions.py -> Includes various functions and a training loop for a Convolutional Variational Autoencoder (CVAE) model.
6) generate_signals.py -> Defines two functions related to generating and saving samples from a trained CVAE model based on a specific age or a range of ages. 


## Overview on PPG signals

PPG (photoplethysmography) signals are a noninvasive monitoring technique used to measure changes in blood volume in subcutaneous vascular structures. For this purpose, a so-called pulse oximeter is used, which is usually attached to a part of the body where blood vessels are close to the surface, such as the finger or the ear. In our case, the device is attached to the finger.
A light emitting diode (LED) sends light to a part of the body with a strong blood supply, covered by a thin layer of skin, while a photodiode measures the amount of light transmitted or reflected.

## Overview on Project aim

This program aims to train a Convolutional Variational Autoencoder (CVAE) model using TensorFlow Keras. The Convolutional Variational Autoencoder is a type of neural network that can generate new data samples after being trained on an input dataset. The CVAE model consists of a neural network architecture that includes an encoder and a decoder.
The encoder takes an input tensor, applies a series of convolutional layers followed by dense layers, and produces the mean and standard deviation of the latent distribution. The decoder takes a sample from the latent distribution, applies a series of transposed convolutional layers followed by dense layers, and produces a reconstructed output tensor.
The program loads the training, validation, and test data from pickle files and converts them into tensors used for model training. The model is trained using the training dataset and evaluated using the validation dataset. The Adam optimizer is used to update the model weights during training
During training, the program displays the current epoch, the loss on the validation dataset, and the time elapsed for the current epoch. An early stopping criterion based on the validation dataset loss is also used to stop training if the loss does not improve for a certain number of consecutive epochs.

# Conditional Variational Autoencoder

Conditional Variational Autoencoder (CVAE) consists of four main components: the encoder, the latent space with condition, and the decoder.

° The encoder plays a crucial role in a Conditional Variational Autoencoder (CVAE) by converting the input data into a meaningful latent represen- tation within the latent space. Its main function is to extract relevant information from the input data and convert it into a compact and ab- stract form. In the context of CVAE, the encoder goes beyond simple data compression and encodes the input data into a probabilistic distri- bution within the latent space. This distribution is characterized by two main vectors: the mean vector and the standard deviation vector. Both vectors are generated by the coder based on the input data and serve as parameters for defining the probability distribution in the latent space. By selecting random points from this distribution, the coder enables the CVAE to produce diverse and realistic outputs during the decoding process.

° The latent space is a compact and disentangled representation of the fea- tures present in the input data, meaning that continuous variations in the latent space correspond to meaningful and semantically significant varia- tions in the generated data. This property allows the CVAE to perform operations such as interpolation and manipulation of features during the data generation process. By navigating in latent space, the CVAE can seamlessly transition and effectively interpolate between different data in- stances. This enables the generation of new data points that have a coher- ent mix of features from the original data. In addition, the disentangled nature of the latent space allows selective manipulation of specific features or attributes, facilitating control over the properties of the generated data.

° Conditioning information is additional data used to condition the data generation process. In a CVAE, the decoder is conditioned on this ad- ditional information when generating the output data. The conditioning information can take different forms depending on the context of the appli- cation. For example, in our project, the conditioning information consists of the label ”age”, but it could be any other information relevant to the data generation process.

° The decoder is responsible for generating the output data based on the conditioned latent representation generated by the encoder and the condi- tioning information provided. After the encoder generates a conditioned latent representation, the decoder takes this representation along with the conditioning information and uses it to generate synthetic output data. The decoder is designed to map the conditioned latent representation back into the output data space with the goal of faithfully reconstructing the original input data. Through learning, the decoder learns to transform the conditioned latent representation into an output that approximates the input data. The decoder is trained to generate data that is consistent with the specified conditioning information. This means that it can gener- ate data that has certain properties or attributes required by the specified condition, which allows more precise control over the generation of the output data.

# Program Structure

The code structure is organized into several sections. Below is a description of the main sections and structures present in the provided code:

° Libraries Import: A set of libraries needed to run the code is imported

° Get Data: Definition of a function that returns several datasets and associated dimensions. The function reads data from files and        converts them to tensors for use with TensorFlow.

° Conv_architectures : convolutional layer configurations that will be used in the definition of the CVAE model.

° Linear_architectures :linear layer configurations that will be used in the definition of the CVAE model.

° CVAE class: This class is a subclass of tf.keras.Model and represents a Conditional Variational Autoencoder model. The class contains methods for building the model, the encoder and decoder, and methods for forward passing through the model. A sample method is also provided for generating samples from the model.

° Loss Function: composed of the log_normal_pdf function calculates the log of the probability density function of a normal distribution and the compute_loss function calculates the loss of the CVAE model. The difference between the decoder output and the original input is calculated, and the log of the probability density of the samples generated by the model is calculated with respect to the normal distribution.

° Definition of a set of functions used to train and operate the model

° function for the generation of synthetic signals


# Optimal Parameters

after implementing a hyperparameter tuning, these are the parameters that optimized the code the best

| parameters                 | values       |
| -------------------------- | ------------ |
| Number of layers encoder   | 9            |
| Number of layers decoder   | 12           |
| Numbers of filters         | 64-128-256   |
| Activation function        | Tanh         |
| Padding                    | Valid        |
| Kernel Size                | 3            |
| Strides                    | 2            |
| Learning Rate              | 0.0001       |
| Optimizer                  | Adam         |


# Results 

| parameters                   | values       |
| ---------------------------- | ------------ |
| Average Spearman Coefficient | 0.6619       |
| Median  Spearman Coefficient | 0.6622       |
| Average MSE                  | 0.8676       |
| Median MSE                   | 0.8831       |



