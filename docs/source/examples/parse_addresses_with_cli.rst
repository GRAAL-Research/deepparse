.. role:: hidden
    :class: hidden-section

Parse Addresses Using Our CLI
*****************************

You can use our cli function to parse rapidly addresses and output them into a file.

.. code-block:: bash

    python3 -m deepparse.cli.parse fasttext ./dataset_path.csv parsed_address.pickle

The cli command will infer the output format (here in a pickle format) and export the parsed addresses next to the
dataset.

You can also specified a device using the ``device`` argument.

.. code-block:: bash

    python3 -m deepparse.cli.parse fasttext ./dataset_path.csv parsed_address.pickle --device 0