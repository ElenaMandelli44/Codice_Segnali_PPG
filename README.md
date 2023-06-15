
# Synthetic_PPG

The purpose of this project is to generate "synthetic PPG signals" after training a Conditional Variational Autoencoder on a pre-existing database consisting of 327,054 previously acquired PPG signals.
Once the neural network was trained, synthetic signals were generated based on specific information provided, such as the hypothetical age of the patient for whom the signals were being generated.

## Prerequisites
The project is coded in Python and makes use of popular scientific packages like numpy, pandas, matplotlib, sklearn, and more. Therefore, it is highly recommended to install Anaconda for smooth execution.

## Overview on PPG signals

PPG (photoplethysmography) signals are a noninvasive monitoring technique used to measure changes in blood volume in subcutaneous vascular structures. For this purpose, a so-called pulse oximeter is used, which is usually attached to a part of the body where blood vessels are close to the surface, such as the finger or the ear. In our case, the device is attached to the finger.
A light emitting diode (LED) sends light to a part of the body with a strong blood supply, covered by a thin layer of skin, while a photodiode measures the amount of light transmitted or reflected.

## Description of the Project

