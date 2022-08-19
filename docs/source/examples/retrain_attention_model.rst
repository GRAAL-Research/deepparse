.. role:: hidden
    :class: hidden-section

Retrain an Attention Mechanism Model
************************************

.. code-block:: python

    import os

    import poutyne

    from deepparse import download_from_public_repository
    from deepparse.dataset_container import PickleDatasetContainer
    from deepparse.parser import AddressParser

First, let's download the train and test data with the new tags, ``"new tags"``, from the public repository.

.. code-block:: python

    saving_dir = "./data"
    file_extension = "p"
    training_dataset_name = "sample_incomplete_data"
    test_dataset_name = "test_sample_data"
    download_from_public_repository(training_dataset_name, saving_dir, file_extension=file_extension)
    download_from_public_repository(test_dataset_name, saving_dir, file_extension=file_extension)

Now let's create a training and test container.

.. code-block:: python

    training_container = PickleDatasetContainer(os.path.join(saving_dir, training_dataset_name + "." + file_extension))
    test_container = PickleDatasetContainer(os.path.join(saving_dir, test_dataset_name + "." + file_extension))

We will retrain the ``FastText`` attention version of our pretrained model.

.. code-block:: python

    model = "bpemb"
    address_parser = AddressParser(model_type=model, device=0, attention_mechanism=True)

Now, let's retrain for ``5`` epochs using a batch size of ``8`` since the data is really small for the example.
Let's start with the default learning rate of ``0.01`` and use a learning rate scheduler to lower the learning rate
as we progress.

.. code-block:: python

    lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch

    logging_path = "./checkpoints"

    address_parser.retrain(
        training_container,
        train_ratio=0.8,
        epochs=5,
        batch_size=8,
        num_workers=2,
        callbacks=[lr_scheduler],
        logging_path=logging_path
    )

Now, let's test our fine-tuned model using the best checkpoint (default parameter).

.. code-block:: python

    address_parser.test(test_container, batch_size=256)
