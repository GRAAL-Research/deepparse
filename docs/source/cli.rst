.. role:: hidden
    :class: hidden-section

CLI
===

You cas use our cli to parsed addresses directly from the command line or download a pre-trained model.

Exporting Format
****************
We support two types of export: CSV and Pickle.

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

Download
--------

.. autofunction:: deepparse.cli.download.main
.. autofunction:: deepparse.cli.parse.main

