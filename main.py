from preprocessing import get_data
from functions import train_model, load_model
from generate_signals import save_samples_from_age_range
from configuration_file import conv_architectures, linear_architectures


def main():
    """
    Execute the main logic of the program.

    This function performs the following steps:
    1. Retrieves data including train, validation, and test datasets, input dimension, latent dimension,
       label dimension, and labels using the 'get_data' function.
    2. Trains or loads a model using the retrieved data, specified architectures, and other parameters
       through the 'train_model' or 'load_model' function.
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
         labels
    ) = get_data(batch_size, sampling_rate=100, working_dir="")

    model = load_model(
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_architectures=conv_architectures,
        linear_architectures=linear_architectures,
        input_dim=input_dim,
    )

    if model is None:
        model = train_model(
            latent_dim=latent_dim,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_dim=label_dim,
            conv_architectures=conv_architectures,
            linear_architectures=linear_architectures,
            batch_size=batch_size,
            input_dim=input_dim,
            epochs=10,
        )
    else:
        print("Using pre-trained model")

    save_samples_from_age_range(model, labels, 18, 99, 1000)


if __name__ == "__main__":
    main()
