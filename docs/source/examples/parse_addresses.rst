.. role:: hidden
    :class: hidden-section

Parse Addresses
***************

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

Let's use the ``BPEmb`` model on a GPU.

.. code-block:: python

    address_parser = AddressParser(model_type="bpemb", device=0)

.. code-block:: python

    parsed_addresses = address_parser(addresses_to_parse[0:300])

    # Print one of the parsed address
    print(parsed_addresses[0])


When parsing addresses, some data quality tests are applied to the dataset.
First, it validates that no addresses to parse are empty.
Second, it validates that no addresses are whitespace-only.
The next two lines are rising a ``DataError``.

.. code-block:: python

    address_parser("")  # Raise an error
    address_parser(" ")  # Raise an error

We can also put our parsed address into a Pandas ``DataFrame`` for analysis. You can choose the fields to use or use the
default one.

.. code-block:: python

    fields = ['StreetNumber', 'StreetName', 'Municipality', 'Province', 'PostalCode']
    parsed_address_data_frame = pd.DataFrame([parsed_address.to_dict(fields=fields) for parsed_address in parsed_addresses],
                                             columns=fields)
