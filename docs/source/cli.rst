.. role:: hidden
    :class: hidden-section

CLI
===

You can use our CLI to parsed addresses directly from the command line, retrain a parsing model or download a
pretrained model.

Parse
*****
The parsing of the addresses to parse ``dataset_path`` is done using the selected ``parsing_model``.
The exported parsed addresses are to be exported in the same directory as the addresses to parse but
given the ``export_file_name`` using the encoding format of the address dataset file. For example,
if the dataset is in a CSV format, the output file format will be a CSV. Moreover, by default,
we log some information (``--log``) such as the parser model name, the parsed dataset path
and the number of parsed addresses. Here is the list of the arguments, their descriptions and default values.
One can use the command ``parse --help`` to output the same description in your command line.

    - ``parsing_model``: The parsing module to use.
    - ``dataset_path``: The path to the dataset file in a pickle (``.p``, ``.pickle`` or ``.pckl``) or CSV format.
    - ``export_file_name``: The filename to use to export the parsed addresses. We will infer the file format base on the file extension. That is, if the file is a pickle (``.p`` or ``.pickle``), we will export it into a pickle file. The supported format are Pickle, CSV and JSON. The file will be exported in the same repositories as the dataset_path. See the doc for more details on the format exporting.
    - ``--device``: The device to use. It can be 'cpu' or a GPU device index such as ``'0'`` or ``'1'``. By default, ``'0'``.
    - ``--batch_size``: The batch size to use to process the dataset. By default, ``32``.
    - ``--path_to_retrained_model``: A path to a retrained model to use for parsing. By default, ``None``.
    - ``--csv_column_name``: The column name to extract address in the CSV. Need to be specified if the provided ``dataset_path`` leads to a CSV file. By default, ``None``.
    - ``--csv_column_separator``: The column separator for the dataset container will only be used if the dataset is a CSV one. By default ``'\t'``.
    - ``--log``: Either or not to log the parsing process into a ``.log`` file exported at the same place as the parsed data using the same name as the export file. The bool value can be (not case sensitive) ``'true/false'``, ``'t/f'``, ``'yes/no'``, ``'y/n'`` or ``'0/1'``. By default, ``True``.

.. autofunction:: deepparse.cli.parse.main

Dataset Format
--------------
For the dataset format see our :class:`~deepparse.dataset_container.DatasetContainer`.

Exporting Format
----------------
We support three types of export formats: CSV, Pickle and JSON.

The first export uses the following pattern column pattern:
``"Address", "First address components class", "Second class", ...``.
Which means the address ``305 rue des Lilas 0 app 2`` will output the table bellow
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

The third export uses a similar approach to the CSV format but uses dictionary-like formatting. Using the
same example will return the following dict ``{"Address": "305 rue des Lilas 0 app 2", "StreetNumber": "305", ...}``.

Retrain
*******

This command allows a user to retrain the ``base_parsing_model`` on the ``train_dataset_path`` dataset.
For the training, the CSV or Pickle dataset is loader in a specific dataloader (see
:class:`~deepparse.dataset_container.DatasetContainer` for more details). We use poutyne's automatic logging
functionalities during training. Thus, it creates an epoch checkpoint and outputs the epoch metrics in a TSV file.
Moreover, we save the best epoch model under the retrain model name (either the default one or a given name using
the ``name_of_the_retrain_parser`` argument). Here is the list of the arguments, their descriptions and default values.
One can use the command ``parse --help`` to output the same description in your command line.

    - ``base_parsing_model``: The parsing module to retrain.
    - ``train_dataset_path``: The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format.
    - ``--train_ratio``: The ratio to use of the dataset for the training. The rest of the data is used for the validation (e.g. a training ratio of 0.8 mean an 80-20 train-valid split) (default is 0.8).
    - ``--batch_size``: The size of the batch (default is ``32``).
    - ``--epochs``: The number of training epochs (default is ``5``).
    - ``--num_workers``: The number of workers to use for the data loader (default is ``1`` worker).
    - ``--learning_rate``: The learning rate (LR) to use for training (default ``0.01``).
    - ``--seed``: The seed to use (default ``42``).
    - ``--logging_path``: The logging path for the checkpoints and the retrained model. Note that training creates checkpoints, and we use Poutyne library that uses the best epoch model and reloads the state if any checkpoints are already there. Thus, an error will be raised if you change the model type. For example, you retrain a FastText model and then retrain a BPEmb in the same logging path directory. By default, the path is ``'./checkpoints'``.
    - ``--disable_tensorboard``: To disable Poutyne automatic Tensorboard monitoring. By default, we disable them (``True``).
    - ``--layers_to_freeze``: Name of the portion of the seq2seq to freeze layers, thus reducing the number of parameters to learn. Default to ``None``.
    - ``--name_of_the_retrain_parser``: Name to give to the retrained parser that will be used when reloaded as the printed name, and to the saving file name. By default ``None``, thus, the default name. See the complete parser retrain method for more details.
    - ``--device``: The device to use. It can be ``'cpu'`` or a GPU device index such as ``'0'`` or ``'1'``. By default ``'0'``.
    - ``--csv_column_names``: The column names to extract address in the CSV. Need to be specified if the provided dataset_path leads to a CSV file. Column names have to be separated by a whitespace. For example, ``--csv_column_names column1 column2``.
    - ``--csv_column_separator``: The column separator for the dataset container will only be used if the dataset is a CSV one. By default ``'\t'``.

.. autofunction:: deepparse.cli.retrain.main

We do not handle the ``seq2seq_params`` and ``prediction_tags`` fine-tuning argument for now.

Download
********
Command to pre-download model weights and requirements. Here is the argument, its description and possible choices,
one can use the command ``parse --help`` to output the same description in your command line.

    - ``model_type``: The parsing module to download. The possible choice are ``'fasttext'``, ``'fasttext-attention'``, ``'fasttext-light'``, ``'bpemb'`` and ``'bpemb-attention'``.

.. autofunction:: deepparse.cli.download.main

