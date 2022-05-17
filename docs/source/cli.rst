.. role:: hidden
    :class: hidden-section

CLI
===

You cas use our cli to parsed addresses directly from the command line or download a pretrained model.

Parse
*****
The parsing of the addresses to parse ``dataset_path`` is done using the selected ``parsing_model``.
The exported parsed addresses are to be exported in the same directory as the addresses to parse but
given the ``export_file_name`` using the encoding format of the address dataset file. For example,
if the dataset is in a CSV format, the output file format will be a CSV. Moreover, by default,
we log some information (``--log``) such as the parser model name, the parsed dataset path
and the number of parsed addresses. Here is the list of the arguments and their descriptions. One can use
the command ``parse --help`` to output the same description in your command line.

- ``parsing_model``: The parsing module to use.
- ``dataset_path``: The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format.
- ``export_file_name``: The filename to use to export the parsed addresses. We will infer the file format base on the file extension. That is, if the file is a pickle (.p or .pickle), we will export it into a pickle file. The supported format are Pickle, CSV and JSON. The file will be exported in the same repositories as the dataset_path. See the doc for more details on the format exporting.
- ``--device``: The device to use. It can be 'cpu' or a GPU device index such as '0' or '1'. By default '0'.
- ``batch_size``: The batch size to use to process the dataset. By default 32.
- ``--path_to_retrained_model``: A path to a retrained model to use for parsing.
- ``--csv_column_name``: The column name to extract address in the CSV. Need to be specified if the provided dataset_path leads to a CSV file.
- ``--csv_column_separator``: The column separator for the dataset container will only be used if the dataset is a CSV one. By default '\t'.
- ``--log``: Either or not to log the parsing process into a `.log` file exported at the same place as the parsed data using the same name as the export file. The bool value can be (not case sensitive) 'true/false', 't/f', 'yes/no', 'y/n' or '0/1'.

.. autofunction:: deepparse.cli.parse.main

Dataset Format
--------------
For the dataset format see our :class:`~deepparse.dataset_container.DatasetContainer`.

Exporting Format
----------------
We support three types of export format: CSV, Pickle and JSON.

The first export uses the following pattern column pattern:
``"Address", "First address components class", "Second class", ...``.
Which mean the address ``305 rue des Lilas 0 app 2`` will output the table bellow
using our default tags:

.. list-table::
        :header-rows: 1

        *   - Address
            - StreetNumber
            - Unit
            - StreetName
            - Orientation
            - Municipality
            - Province
            - Postal Code
            - GeneralDelivery
        *   - 305 rue des Lilas 0 app 2
            - 305
            - app 2
            - rue des lilas
            - o
            - None
            - None
            - None
            - None

The second export uses a similar approach but using tuples and list. Using the same example will return the following
tuple ``("305 rue des Lilas 0 app 2", [("305", "StreetNumber"), ("rue des lilas", "StreetName"), ...])``.

The third export uses a similar approach as the CSV format but instead use a dictionary-like formatting. Using the
same example will return the following dict ``{"Address": "305 rue des Lilas 0 app 2", "StreetNumber": "305", ...}``.

Download
********
Command to pre-download model weights and requirements. Here is the argument, its description and possible choices,
one can use the command ``parse --help`` to output the same description in your command line.

- ``model_type``: The parsing module to download. The possible choice are ``fasttext``, ``fasttext-attention``,
``fasttext-light``, ``bpemb`` and ``bpemb-attention``.


.. autofunction:: deepparse.cli.download.main

