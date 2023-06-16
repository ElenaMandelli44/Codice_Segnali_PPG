import main
import analisi
import pickle
import ipdb
import numpy as np
from tensorflow import keras
import random


def test():
  
  """
      test() function, call the get_data() function of the main module 
  """
    batch_size = 64
    (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        latent_dim,
        label_dim,
        input_dim,
        labels,
    ) = main.get_data(batch_size)
    model = main.train_or_load_model(
        epochs=100,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_architectures=main.conv_architectures,
        linear_architectures=main.linear_architectures,
        batch_size=batch_size,
        input_dim=input_dim,
    )
    
    
    """   #Checks if the model returned by the train_or_load_model() 
    function is an instance of TensorFlow's keras.Model
    """
    assert isinstance(model, keras.Model)
    
    
    """"  Check that the function returns a tuple with two elements, 
    and that the sizes of the generated samples are consistent.   """"
    samples = main.generate_samples_from_age(model, labels, random.randint(18, 80), 20)
    assert len(samples) == 2
    assert samples[0].shape[0] == samples[1].shape[0]

    
    main.save_samples_from_age_range(model, labels, 18, 99, 1000)
    with open("generated_samples_18_99.pickle", "rb") as file:
        loaded_data = pickle.load(file)

    """
    Call the main module's save_samples_from_age_range() function to generate and save 
    a large number of samples for a specific age range. Verify that the generated samples file was created correctly.
    """
    assert len(loaded_data) == 2
    
    """
     Load data from the generated samples file and verify that the loaded data has the correct structure.
    """
    assert loaded_data[0].shape[0] == loaded_data[1].shape[0]
    
    """
    Generate two arrays of random values and check if the compute_metrics() function of the analysis module returns the expected results.
    """
    rnd_val1 = np.random.rand(100)
    rnd_val2 = np.random.rand(100)
    assert analisi.compute_metrics(rnd_val1, rnd_val1) == (1.0, 0.0)
    assert analisi.compute_metrics(rnd_val1, rnd_val2)[1] > 0.0
