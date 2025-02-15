.. role:: hidden
    :class: hidden-section

Parse Addresses Using A URI
***************************

.. code-block:: python

    import pandas as pd

    from deepparse import download_from_public_repository
    from deepparse.dataset_container import PickleDatasetContainer
    from deepparse.parser import AddressParser

Here is an example on how to parse multiple addresses. First, let's download the train and test data from the public repository.

.. code-block:: python

    saving_dir = "./data"
    file_extension = "p"
    test_dataset_name = "predict"
    download_from_public_repository(test_dataset_name, saving_dir, file_extension=file_extension)

Now let's load the dataset using one of our dataset container

.. code-block:: python

    addresses_to_parse = PickleDatasetContainer("./data/predict.p", is_training_container=False)

# Let's use the ``FastText`` model on a GPU.

.. code-block:: python

    path_to_your_uri = "s3://<path_to_your_bucket>/fasttext.ckpt"
    address_parser = AddressParser(model_type="fasttext", device=0, path_to_retrained_model=path_to_your_uri)


.. code-block:: python

    parsed_addresses = address_parser(test_data[0:300])

    # Print one of the parsed address
    print(parsed_addresses[0])
