def main():
    
    """
    Execute the main logic of the program.

    This function performs the following steps:
    1. Retrieves data including train, validation, and test datasets, input dimension, latent dimension,
       label dimension, and labels using the 'get_data' function.
    2. Trains or loads a model using the retrieved data, specified architectures, and other parameters
       through the 'train_or_load_model' function.
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
    ) = get_data(batch_size)
    model = train_or_load_model(
        epochs=300,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        latent_dim=latent_dim,
        label_dim=label_dim,
        conv_architectures=conv_architectures,
        linear_architectures=linear_architectures,
        batch_size=batch_size,
        input_dim=input_dim,
    )
    save_samples_from_age_range(model, labels, 18, 99, 1000)


if __name__ == "__main__":
    main()

