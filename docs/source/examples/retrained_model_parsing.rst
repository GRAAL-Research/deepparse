.. role:: hidden
    :class: hidden-section

Use a Retrained Model to Parse Addresses
****************************************

.. code-block:: python

    import os

    from deepparse import download_from_public_repository
    from deepparse.dataset_container import PickleDatasetContainer
    from deepparse.parser import AddressParser

Here is an example on how to parse multiple addresses using a retrained model.
First, let's download the train and test data from the public repository.

.. code-block:: python

    data_saving_dir = "./data"
    file_extension = "p"
    test_dataset_name = "predict"
    download_from_public_repository(test_dataset_name, data_saving_dir, file_extension=file_extension)

Now let's load the dataset using one of our dataset container.

.. code-block:: python

    addresses_to_parse = PickleDatasetContainer("./data/predict.p", is_training_container=False)


Let's download a ``BPEmb`` retrained model create just for this example, but you can also use one of yours.

.. code-block:: python

    model_saving_dir = "./retrained_models"
    retrained_model_name = "retrained_light_bpemb_address_parser"
    model_file_extension = "ckpt"
    download_from_public_repository(retrained_model_name, model_saving_dir, file_extension=model_file_extension)

    address_parser = AddressParser(
        model_type="bpemb", device=0,
        path_to_retrained_model=os.path.join(model_saving_dir, retrained_model_name + "." + model_file_extension)
    )

We can now parse some addresses

.. code-block:: python

    parsed_addresses = address_parser(addresses_to_parse[0:300])
