.. role:: hidden
    :class: hidden-section

Getting Started
===============

.. code-block:: python

   from deepparse.parser import AddressParser
   from deepparse.dataset_container import CSVDatasetContainer

   address_parser = AddressParser(model_type="bpemb", device=0)

   # you can parse one address
   parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

   # or multiple addresses
   parsed_address = address_parser(["350 rue des Lilas Ouest Québec Québec G1L 1B6",
        "350 rue des Lilas Ouest Québec Québec G1L 1B6"])

   # or multinational addresses
   # Canada, US, Germany, UK and South Korea
   parsed_address = address_parser(
       ["350 rue des Lilas Ouest Québec Québec G1L 1B6", "777 Brockton Avenue, Abington MA 2351",
        "Ansgarstr. 4, Wallenhorst, 49134", "221 B Baker Street", "서울특별시 종로구 사직로3길 23"])

   # you can also get the probability of the predicted tags
   parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6",
        with_prob=True)

   # Print the parsed address
   print(parsed_address)

   # or using one of our dataset container
   addresses_to_parse = CSVDatasetContainer("./a_path.csv", column_names=["address_column_name"],
                                            is_training_container=False)
   address_parser(addresses_to_parse)

The default predictions tags are the following

    - ``"StreetNumber"``: for the street number,
    - ``"StreetName"``: for the name of the street,
    - ``"Unit"``: for the unit (such as apartment),
    - ``"Municipality"``: for the municipality,
    - ``"Province"``: for the province or local region,
    - ``"PostalCode"``: for the postal code,
    - ``"Orientation"``: for the street orientation (e.g. west, east),
    - ``"GeneralDelivery"``: for other delivery information.

Parse Addresses From the Command Line
*************************************

You can also use our cli to parse addresses using:

.. code-block:: sh

    parse <parsing_model> <dataset_path> <export_file_name>

Parse Addresses Using Your Own Retrained Model
**********************************************

See `here <https://github.com/GRAAL-Research/deepparse/blob/main/examples/retrained_model_parsing.py>`__ for a complete example.

.. code-block:: python

    address_parser = AddressParser(
        model_type="bpemb", device=0, path_to_retrained_model="path/to/retrained/bpemb/model.p")

    address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

Retrain a Model
***************
See `here <https://github.com/GRAAL-Research/deepparse/blob/main/examples/fine_tuning.py>`__ for a complete example
using Pickle and `here <https://github.com/GRAAL-Research/deepparse/blob/main/examples/fine_tuning_with_csv_dataset.py>`__
for a complete example using CSV.

.. code-block:: python

    address_parser.retrain(training_container, train_ratio=0.8, epochs=5, batch_size=8)

One can also freeze some layers to speed up the training using the ``layers_to_freeze`` parameter.

.. code-block:: python

    address_parser.retrain(training_container, train_ratio=0.8, epochs=5, batch_size=8, layers_to_freeze="seq2seq")


Or you can also give a specific name to the retrained model. This name will be use as the model name (for print and
class name) when reloading it.

.. code-block:: python

    address_parser.retrain(training_container, train_ratio=0.8, epochs=5, batch_size=8, name_of_the_retrain_parser="MyNewParser")




Retrain a Model With an Attention Mechanism
*******************************************
See `here <https://github.com/GRAAL-Research/deepparse/blob/main/examples/retrain_attention_model.py>`__ for a complete example.

.. code-block:: python

    # We will retrain the fasttext version of our pretrained model.
    address_parser = AddressParser(model_type="fasttext", device=0, attention_mechanism=True)

    address_parser.retrain(training_container, train_ratio=0.8, epochs=5, batch_size=8)


Retrain a Model With New Tags
*****************************
See `here <https://github.com/GRAAL-Research/deepparse/blob/main/examples/retrain_with_new_prediction_tags.py>`__ for a complete example.

.. code-block:: python

    address_components = {"ATag":0, "AnotherTag": 1, "EOS": 2}
    address_parser.retrain(training_container, train_ratio=0.8, epochs=1, batch_size=128, prediction_tags=address_components)


Retrain a Seq2Seq Model From Scratch
************************************

See  `here <https://github.com/GRAAL-Research/deepparse/blob/main/examples/retrain_with_new_seq2seq_params.py>`__ for
a complete example.

.. code-block:: python

    seq2seq_params = {"encoder_hidden_size": 512, "decoder_hidden_size": 512}
    address_parser.retrain(training_container, train_ratio=0.8, epochs=1, batch_size=128, seq2seq_params=seq2seq_params)


Download Our Models
*******************

Here are the URLs to download our pretrained models directly
    - `FastText <https://graal.ift.ulaval.ca/public/deepparse/fasttext.ckpt>`__,
    - `FastTextAttention <https://graal.ift.ulaval.ca/public/deepparse/fasttext_attention.ckpt>`__,
    - `BPEmb <https://graal.ift.ulaval.ca/public/deepparse/bpemb.ckpt>`__,
    - `BPEmbAttention <https://graal.ift.ulaval.ca/public/deepparse/bpemb_attention.ckpt>`__,
    - `FastText Light <https://graal.ift.ulaval.ca/public/deepparse/fasttext.magnitude.gz>`__ (using `Magnitude Light <https://github.com/davebulaval/magnitude-light>`__),.

Or you can use our cli to download our pretrained models directly using:

.. code-block:: sh

    download_model <model_name>

