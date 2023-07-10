from argparse import ArgumentParser

from .tools import wrap, bool_parse
from .. import MODEL_MAPPING_CHOICES


def add_base_parsing_model_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "base_parsing_model",
        choices=MODEL_MAPPING_CHOICES,
        help=wrap("The base parsing module to use for retraining."),
    )


def add_device_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        help=wrap("The device to use. It can be 'cpu' or a GPU device index such as '0' or '1'. By default '0'."),
        type=str,
        default="0",
    )


def add_csv_column_name_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--csv_column_name",
        help=wrap(
            "The column name to extract address in the CSV. Need to be specified if the provided dataset_path "
            "leads to a CSV file."
        ),
        type=str,
        default=None,
    )


def add_csv_column_names_arg(parser: ArgumentParser) -> None:
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


def add_seed_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        help=wrap("The seed to use to make the sampling deterministic (default 42)."),
        type=int,
        default=42,
    )


def add_csv_column_separator_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--csv_column_separator",
        help=wrap(
            "The column separator for the dataset container will only be used if the dataset is a CSV one."
            " By default '\t'."
        ),
        default="\t",
    )


def add_log_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--log",
        help=wrap(
            "Either or not to log the parsing process into a '.log' file exported at the same place as the "
            "parsed data using the same name as the export file. "
            "The bool value can be (not case sensitive) 'true/false', 't/f', 'yes/no', 'y/n' or '0/1'."
        ),
        type=bool_parse,
        default="True",
    )


def add_cache_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="To change the default cache directory (default to None e.g. default path).",
    )


def add_batch_size_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--batch_size",
        help=wrap("The size of the batch (default is 32)."),
        type=int,
        default=32,
    )


def add_path_to_retrained_model_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--path_to_retrained_model",
        help=wrap("A path to a retrained model to use. It can be an S3-URI."),
        type=str,
        default=None,
    )


def add_num_workers_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--num_workers",
        help=wrap("The number of workers to use for the data loader (default is 1 worker)."),
        type=int,
        default=1,
    )
