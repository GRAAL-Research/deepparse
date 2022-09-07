.. role:: hidden
    :class: hidden-section

Parse Addresses Using Our CLI
*****************************

You can use our cli function to parse addresses rapidly and output them into a file.

.. code-block:: bash

    parse fasttext ./dataset_path.csv parsed_address.pickle

The cli command will infer the output format (here in a pickle format) and export the parsed addresses next to the
dataset.

You can also specify a device using the optional ``device`` argument.

.. code-block:: bash

    parse fasttext ./dataset_path.csv parsed_address.pickle --device 0

Or use one of your retrained models using the optional ``path_to_retrained_model`` argument.

.. code-block:: bash

    parse fasttext ./dataset_path.csv parsed_address.pickle --path_to_retrained_model ./path
