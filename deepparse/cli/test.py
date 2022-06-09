import argparse
import logging
import sys

import pandas as pd

from deepparse.cli.tools import (
    is_csv_path,
    is_pickle_path,
    wrap,
    bool_parse,
    attention_model_type_handling,
    generate_export_path,
    replace_path_extension,
)
from deepparse.dataset_container import CSVDatasetContainer, PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args=None) -> None:
    # pylint: disable=too-many-locals, too-many-branches
    """
    CLI function to rapidly test an address parser on test data using the same argument as the
    :meth:`~AddressParser.test` method (with the same default values) except for the callbacks.
    The results will be logged in a CSV file next to the test dataset.


    Examples of usage:

    .. code-block:: sh

        test fasttext ./test_dataset_path.csv

    Modifying testing parameters

    .. code-block:: sh

        test bpemb ./test_dataset_path.csv --batch_size 128 --logging_path "./logging_test"

    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]

    parsed_args = get_args(args)

    test_dataset_path = parsed_args.test_dataset_path
    if is_csv_path(test_dataset_path):
        csv_column_names = parsed_args.csv_column_names
        if csv_column_names is None:
            raise ValueError(
                "To use a CSV dataset to test on, you need to specify the 'csv_column_names' argument to provide the"
                " column name to extract address."
            )
        csv_column_separator = parsed_args.csv_column_separator
        testing_data = CSVDatasetContainer(
            test_dataset_path,
            column_names=csv_column_names,
            separator=csv_column_separator,
            is_training_container=True,
        )
    elif is_pickle_path(test_dataset_path):
        testing_data = PickleDatasetContainer(test_dataset_path, is_training_container=True)
    else:
        raise ValueError("The test dataset path argument is not a CSV or a pickle file.")

    device = parsed_args.device

    if "cpu" not in device:
        device = int(device)
    parser_args = {"device": device}

    path_to_retrained_model = parsed_args.path_to_retrained_model
    if path_to_retrained_model is not None:
        parser_args.update({"path_to_retrained_model": path_to_retrained_model})

    base_parsing_model = parsed_args.base_parsing_model
    parser_args_update_args = attention_model_type_handling(base_parsing_model)
    parser_args.update(**parser_args_update_args)

    address_parser = AddressParser(**parser_args)

    batch_size = parsed_args.batch_size
    num_workers = parsed_args.num_workers
    seed = parsed_args.seed
    parser_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }

    results_export_path = generate_export_path(test_dataset_path, f"{str(address_parser)}_testing.tsv")
    if parsed_args.log:
        logging_export_path = replace_path_extension(results_export_path, ".log")
        logging.basicConfig(
            filename=logging_export_path, format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )

        text_to_log = f"Testing results on dataset file {test_dataset_path} using the parser {str(address_parser)}."
        logging.info(text_to_log)

    results = address_parser.test(test_dataset_container=testing_data, **parser_args)

    pd.DataFrame(results, index=[0]).to_csv(results_export_path, index=False, sep="\t")
    if parsed_args.log:
        text_to_log = (
            f"Testing on the dataset file {test_dataset_path} is finished. The results are logged in "
            f"the CSV file at {results_export_path}."
        )
        logging.info(text_to_log)


def get_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for the cli."""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "base_parsing_model",
        choices=[
            "fasttext",
            "fasttext-attention",
            "fasttext-light",
            "bpemb",
            "bpemb-attention",
        ],
        help=wrap("The base parsing module to use for testing."),
    )

    parser.add_argument(
        "test_dataset_path",
        help=wrap("The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format."),
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
        help=wrap("A path to a retrained model to use for testing."),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        help=wrap("The size of the batch (default is 32)."),
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        help=wrap("The number of workers to use for the data loader (default is 1 worker)."),
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        help=wrap("The seed to use to make the sampling deterministic (default 42)."),
        type=int,
        default=42,
    )

    parser.add_argument(
        "--csv_column_names",
        help=wrap(
            "The column names to extract address and tags in the CSV. Need to be specified if the provided "
            "dataset_path leads to a CSV file. Column names have to be separated by a whitespace. For"
            "example, --csv_column_names column1 column2. By default, None."
        ),
        default=None,
        nargs=2,
        type=str,
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
