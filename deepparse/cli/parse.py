import argparse
import logging
import sys
from functools import partial

from .parser_arguments_adder import (
    add_device_arg,
    add_csv_column_separator_arg,
    add_csv_column_name_arg,
    add_log_arg,
    add_cache_dir_arg,
    add_batch_size_arg,
    add_path_to_retrained_model_arg,
    choices,
)
from .tools import (
    is_csv_path,
    is_pickle_path,
    to_csv,
    to_pickle,
    generate_export_path,
    wrap,
    is_json_path,
    to_json,
    replace_path_extension,
    attention_model_type_handling,
    data_container_factory,
)
from ..parser import AddressParser


def main(args=None) -> None:
    # pylint: disable=too-many-locals, too-many-branches
    """
    CLI function to rapidly parse an addresses dataset and output it in another file.

    Examples of usage:

    .. code-block:: sh

        parse fasttext ./dataset_path.csv parsed_address.pickle

    Using a gpu device

    .. code-block:: sh

        parse fasttext ./dataset_path.csv parsed_address.p --device 0

    Using a CSV dataset

    .. code-block:: sh

        parse fasttext ./dataset.csv parsed_address.pckl --path_to_retrained_model ./path

    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]

    parsed_args = get_args(args)

    dataset_path = parsed_args.dataset_path
    csv_column_separator = parsed_args.csv_column_separator
    addresses_to_parse = data_container_factory(
        dataset_path=dataset_path,
        trainable_dataset=False,
        csv_column_separator=csv_column_separator,
        csv_column_name=parsed_args.csv_column_name,
    )

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
    parser_args = {"device": device, "cache_dir": parsed_args.cache_dir}

    parser_args_update_args = attention_model_type_handling(parsing_model)
    parser_args.update(**parser_args_update_args)

    if path_to_retrained_model is not None:
        parser_args.update({"path_to_retrained_model": path_to_retrained_model})

    address_parser = AddressParser(**parser_args)

    if parsed_args.log:
        logging_export_path = replace_path_extension(export_path, ".log")
        logging.basicConfig(
            filename=logging_export_path, format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )

        text_to_log = f"Parsing dataset file {dataset_path} using the parser {str(address_parser)}"
        logging.info(text_to_log)

    parsed_address = address_parser(addresses_to_parse, batch_size=parsed_args.batch_size)

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
        choices=choices,
        help=wrap("The parsing module to use."),
    )

    parser.add_argument(
        "dataset_path",
        help=wrap("The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format."),
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

    add_path_to_retrained_model_arg(parser)

    add_device_arg(parser)

    add_batch_size_arg(parser)

    add_csv_column_name_arg(parser)

    add_csv_column_separator_arg(parser)

    add_log_arg(parser)

    add_cache_dir_arg(parser)

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
