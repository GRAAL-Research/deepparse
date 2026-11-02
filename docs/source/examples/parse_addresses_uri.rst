.. role:: hidden
    :class: hidden-section

Parse Addresses Using A URI
***************************

.. code-block:: python

    import pandas as pd

    from deepparse import download_worldwide_addresses_sample
    from deepparse.dataset_container import ListDatasetContainer
    from deepparse.parser import AddressParser

Here is an example on how to parse multiple addresses. First, let's download a small sample of Canadian addresses from the
`worldwide-addresses dataset <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_. We only keep the
addresses (not the tags).

.. code-block:: python

    addresses = [address for address, _ in download_worldwide_addresses_sample(country_code="ca", num_samples=300)]

Now let's load the dataset using one of our dataset container

.. code-block:: python

    addresses_to_parse = ListDatasetContainer(addresses, is_training_container=False)

# Let's use the ``FastText`` model on a GPU.

.. code-block:: python

    path_to_your_uri = "s3://<path_to_your_bucket>/fasttext.ckpt"
    address_parser = AddressParser(model_type="fasttext", device=0, path_to_retrained_model=path_to_your_uri)


.. code-block:: python

    parsed_addresses = address_parser(addresses_to_parse[0:300])

    # Print one of the parsed address
    print(parsed_addresses[0])
