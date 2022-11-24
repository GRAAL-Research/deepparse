import argparse
import logging
import sys

import pandas as pd

from .parser_arguments_adder import (
    add_csv_column_separator_arg,
    add_log_arg,
    add_cache_dir_arg,
    add_seed_arg,
    add_device_arg,
    add_batch_size_arg,
    add_path_to_retrained_model_arg,
    add_base_parsing_model_arg,
    add_num_workers_arg,
    add_csv_column_names_arg,
)
from .tools import (
    wrap,
    attention_model_type_handling,
    generate_export_path,
    replace_path_extension,
    data_container_factory,
)
from ..parser import AddressParser


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

    testing_data = data_container_factory(
        dataset_path=test_dataset_path,
        trainable_dataset=True,
        csv_column_separator=parsed_args.csv_column_separator,
        csv_column_names=parsed_args.csv_column_names,
    )

    device = parsed_args.device

    if "cpu" not in device:
        device = int(device)
    parser_args = {"device": device, "cache_dir": parsed_args.cache_dir}

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
    test_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }

    results_export_path = generate_export_path(test_dataset_path, f"{address_parser}_testing.tsv")
    if parsed_args.log:
        logging_export_path = replace_path_extension(results_export_path, ".log")
        logging.basicConfig(
            filename=logging_export_path, format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )

        text_to_log = f"Testing results on dataset file {test_dataset_path} using the parser {address_parser}."
        logging.info(text_to_log)

    results = address_parser.test(test_dataset_container=testing_data, **test_arguments)

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

    add_base_parsing_model_arg(parser)

    parser.add_argument(
        "test_dataset_path",
        help=wrap("The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format."),
        type=str,
    )

    add_path_to_retrained_model_arg(parser)

    add_batch_size_arg(parser)

    add_num_workers_arg(parser)

    add_seed_arg(parser)

    add_device_arg(parser)

    add_csv_column_names_arg(parser)

    add_csv_column_separator_arg(parser)

    add_log_arg(parser)

    add_cache_dir_arg(parser)

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
