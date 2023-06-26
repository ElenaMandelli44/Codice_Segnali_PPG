import pickle
import pandas as pd
import numpy as np


# Test della funzione load_data
file_path = "data.pickle"
labels, samples = load_data(file_path)
print("Loaded labels shape:", labels.shape)
print("Loaded samples shape:", samples.shape)
