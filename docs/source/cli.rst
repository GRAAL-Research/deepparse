.. role:: hidden
    :class: hidden-section

CLI
===

You cas use our cli to parsed addresses directly from the command line or download a pre-trained model.

Parse
*****
The parsing of the addresses to parse ``dataset_path`` is done using the selected ``parsing_model``.
The exported parsed addresses are to be exported in the same directory as the addresses to parse but
given the ``export_file_name`` using the encoding format of the address dataset file. For example,
if the dataset is in a CSV format, the output file format will be a CSV. Moreover, by default,
we log some information (``--log``) such as the parser model name, the parsed dataset path
and the number of parsed addresses.

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

.. autofunction:: deepparse.cli.download.main

