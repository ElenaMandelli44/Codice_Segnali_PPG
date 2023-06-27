import os
from preprocessing import get_data
from functions import train_model, load_model
from generate_signals import save_samples_from_age_range
from class_CVAE import conv_architectures, linear_architectures


def main():
    """
    Execute the main logic of the program.

    This function performs the following steps:
    1. Retrieves data including train, validation, and test datasets, input dimension, latent dimension,
       label dimension, and labels using the 'get_data' function.
    2. Checks if a pre-trained model exists. If yes, loads the model; otherwise, trains a new model.
    3. Saves samples from a specified age range using the trained model and labels.
    """

    batch_size = 64
    (
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        latent_dim,
        label_dim,
        labels,
    ) = get_data(batch_size, sampling_rate=100, working_dir=None)
    
    train_model_flag = not os.path.exists("trained_model.index")

    if train_model_flag:
        # Train the model
        model = train_model(
            latent_dim=latent_dim,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
            label_dim=label_dim,
            conv_architectures=conv_architectures,
            linear_architectures=linear_architectures,
            batch_size=batch_size,
            input_dim=input_dim,
            epochs=3,
        )
        model.save_weights("trained_model")
    else:
        # Load a pre-trained model
        model = load_model(
            latent_dim=latent_dim,
            label_dim=label_dim,
            conv_architectures=conv_architectures,
            linear_architectures=linear_architectures,
            input_dim=input_dim,
        )
    
    save_samples_from_age_range(model, labels, 18, 99, 1000)


if __name__ == "__main__":
    main()


