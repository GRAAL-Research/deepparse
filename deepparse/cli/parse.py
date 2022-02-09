import argparse
from functools import partial

from deepparse.cli.tools import is_csv_path, is_pickle_path, to_csv, to_pickle
from deepparse.dataset_container import CSVDatasetContainer, PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args: argparse.Namespace) -> None:
    """
    cli function to rapidly parse an addresses dataset and output it in another file.

    Examples of usage:

    .. code-block:: sh

        python3 -m deepparse.cli.parse fasttext ./dataset_path.csv parsed_address.pickle

    Using a gpu device

    .. code-block:: sh

        python3 -m deepparse.cli.parse fasttext ./dataset_path.csv parsed_address.pickle --device 0

    Using a CSV dataset

    .. code-block:: sh

        python3 -m deepparse.cli.parse fasttext ./dataset.csv parsed_address.pickle --path_to_retrained_model ./path

    """

    dataset_path = args.dataset_path
    if is_csv_path(dataset_path):
        csv_column_name = args.csv_column_name
        if csv_column_name is None:
            raise ValueError(
                "For a CSV dataset path, you need to specify the 'csv_column_name' argument to provide the"
                " column name to extract address."
            )
        csv_column_separator = args.csv_column_separator
        addresses_to_parse = CSVDatasetContainer(
            dataset_path, column_names=[csv_column_name], separator=csv_column_separator, is_training_container=False
        )
    elif is_pickle_path(dataset_path):
        addresses_to_parse = PickleDatasetContainer(dataset_path, is_training_container=False)
    else:
        raise ValueError("The dataset path argument is not a CSV or pickle file.")

    export_file_name = args.export_file_name
    export_path = dataset_path

    if is_csv_path(export_file_name):
        export_fn = partial(to_csv, export_path=export_path, sep=csv_column_separator)
    elif is_pickle_path(export_file_name):
        export_fn = partial(to_pickle, export_path=export_path)
    else:
        ValueError("We do not support this type of export.")

    parsing_model = args.parsing_model
    device = args.device
    path_to_retrained_model = args.path_to_retrained_model

    if "cpu" not in device:
        device = int(device)
    parser_args = {"device": device}
    if "-attention" in parsing_model:
        parser_args.update({"attention_mechanism": True})
        parsing_model = parsing_model.strip("attention").strip("-")
    parser_args.update({"model_type": parsing_model})

    if path_to_retrained_model is not None:
        parser_args.update({"path_to_retrained_model": path_to_retrained_model})

    address_parser = AddressParser(**parser_args)

    parsed_address = address_parser(addresses_to_parse)

    export_fn(parsed_address)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "parsing_model",
        choices=[
            "fasttext",
            "fasttext-attention",
            "fasttext-light",
            "bpemb",
            "bpemb-attention",
        ],
        help="The parsing module to use.",
    )

    parser.add_argument(
        "dataset_path", help="The path to the dataset file in a pickle (.p or .pickle) or CSV format.", type=str
    )

    parser.add_argument(
        "export_file_name",
        help="The file name to use for the export of the parsed addresses. We will infer the file format base on the "
        "file extension. That is, if the file is a pickle (.p or .pickle), we will export it into a pickle file."
        "The file will be exported in the same repositories as the dataset_path.",
        type=str,
    )

    parser.add_argument(
        "--device",
        help="The device to use. It can be 'cpu' or a gpu device index such as '0' or '1'. By default '0'.",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--path_to_retrained_model", help="A path to a retrained model to use for parsing.", type=str, default=None
    )

    parser.add_argument(
        "--csv_column_name",
        help="The column name to extract address in the CSV. Need to be specified if the provided dataset_path is "
        "leading to a CSV file.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--csv_column_separator",
        help="The column separator to use for the dataset container. By default '\t'.",
        default="\t",
    )

    args_parser = parser.parse_args()

    main(args_parser)
