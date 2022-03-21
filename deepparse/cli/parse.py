import argparse
import logging
import sys
from functools import partial

from deepparse.cli.tools import (
    is_csv_path,
    is_pickle_path,
    to_csv,
    to_pickle,
    generate_export_path,
    wrap,
    is_json_path,
    to_json,
    bool_parse,
    replace_path_extension,
)
from deepparse.dataset_container import CSVDatasetContainer, PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args=None) -> None:
    # pylint: disable=too-many-locals, too-many-branches
    """
    CLI function to rapidly parse an addresses dataset and output it in another file.

    Examples of usage:

    .. code-block:: sh

        parse fasttext ./dataset_path.csv parsed_address.pickle

    Using a gpu device

    .. code-block:: sh

        parse fasttext ./dataset_path.csv parsed_address.pickle --device 0

    Using a CSV dataset

    .. code-block:: sh

        parse fasttext ./dataset.csv parsed_address.pickle --path_to_retrained_model ./path

    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]

    parsed_args = get_args(args)

    dataset_path = parsed_args.dataset_path
    if is_csv_path(dataset_path):
        csv_column_name = parsed_args.csv_column_name
        if csv_column_name is None:
            raise ValueError(
                "For a CSV dataset path, you need to specify the 'csv_column_name' argument to provide the"
                " column name to extract address."
            )
        csv_column_separator = parsed_args.csv_column_separator
        addresses_to_parse = CSVDatasetContainer(
            dataset_path, column_names=[csv_column_name], separator=csv_column_separator, is_training_container=False
        )
    elif is_pickle_path(dataset_path):
        addresses_to_parse = PickleDatasetContainer(dataset_path, is_training_container=False)
    else:
        raise ValueError("The dataset path argument is not a CSV or pickle file.")

    export_filename = parsed_args.export_filename
    export_path = generate_export_path(dataset_path, export_filename)

    if is_csv_path(export_filename):
        export_fn = partial(to_csv, export_path=export_path, sep=csv_column_separator)
    elif is_pickle_path(export_filename):
        export_fn = partial(to_pickle, export_path=export_path)
    elif is_json_path(export_filename):
        export_fn = partial(to_json, export_path=export_path)
    else:
        raise ValueError("We do not support this type of export.")

    parsing_model = parsed_args.parsing_model
    device = parsed_args.device
    path_to_retrained_model = parsed_args.path_to_retrained_model

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

    if parsed_args.log:
        logging_export_path = replace_path_extension(export_path, ".log")
        logging.basicConfig(
            filename=logging_export_path, format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )

        text_to_log = f"Parsing dataset file {dataset_path} using the parser {address_parser}"
        logging.info(text_to_log)

    parsed_address = address_parser(addresses_to_parse)

    export_fn(parsed_address)

    print(f"{len(addresses_to_parse)} addresses have been parsed.")

    if parsed_args.log:
        text_to_log = (
            f"{len(addresses_to_parse)} addresses have been parsed.\n"
            f"The parsed addresses are outputted here: {export_path}"
        )
        logging.info(text_to_log)


def get_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for the cli."""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "parsing_model",
        choices=[
            "fasttext",
            "fasttext-attention",
            "fasttext-light",
            "bpemb",
            "bpemb-attention",
        ],
        help=wrap("The parsing module to use."),
    )

    parser.add_argument(
        "dataset_path",
        help=wrap("The path to the dataset file in a pickle (.p or .pickle), CSV format or JSON format."),
        type=str,
    )

    parser.add_argument(
        "export_filename",
        help=wrap(
            "The filename to use to export the parsed addresses. We will infer the file format base on the "
            "file extension. That is, if the file is a pickle (.p or .pickle), we will export it into a pickle file. "
            "The supported format are Pickle, CSV and JSON. "
            "The file will be exported in the same repositories as the dataset_path. "
            "See the doc for more details on the format exporting."
        ),
        type=str,
    )

    parser.add_argument(
        "--device",
        help=wrap("The device to use. It can be 'cpu' or a GPU device index such as '0' or '1'. By default '0'."),
        type=str,
        default="0",
    )

    parser.add_argument(
        "--path_to_retrained_model",
        help=wrap("A path to a retrained model to use for parsing."),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--csv_column_name",
        help=wrap(
            "The column name to extract address in the CSV. Need to be specified if the provided dataset_path "
            "leads to a CSV file."
        ),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--csv_column_separator",
        help=wrap(
            "The column separator for the dataset container will only be used if the dataset is a CSV one."
            " By default '\t'."
        ),
        default="\t",
    )

    parser.add_argument(
        "--log",
        help=wrap(
            "Either or not to log the parsing process into a `.log` file exported at the same place as the "
            "parsed data using the same name as the export file. "
            "The bool value can be (not case sensitive) 'true/false', 't/f', 'yes/no', 'y/n' or '0/1'."
        ),
        type=bool_parse,
        default="True",
    )

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
