import argparse
import sys
from typing import Dict

from .parser_arguments_adder import (
    add_seed_arg,
    add_batch_size_arg,
    add_base_parsing_model_arg,
    add_num_workers_arg,
    add_device_arg,
    add_csv_column_separator_arg,
    add_cache_dir_arg,
    add_csv_column_names_arg,
)
from .tools import (
    wrap,
    bool_parse,
    attention_model_type_handling,
    data_container_factory,
)
from ..parser import AddressParser

_retrain_parameters = [
    "train_ratio",
    "batch_size",
    "epochs",
    "num_workers",
    "learning_rate",
    "seed",
    "logging_path",
    "disable_tensorboard",
    "layers_to_freeze",
    "name_of_the_retrain_parser",
]


def parse_retrained_arguments(parsed_args) -> Dict:
    dict_parsed_args = vars(parsed_args)
    parsed_retain_arguments = {}

    for retrain_parameter in _retrain_parameters:
        value = dict_parsed_args.get(retrain_parameter)
        parsed_retain_arguments.update({retrain_parameter: value})

    return parsed_retain_arguments


def main(args=None) -> None:
    # pylint: disable=too-many-locals, too-many-branches
    """
    CLI function to rapidly retrain an addresses parser and saves it. One can retrain a base pretrained model
    using most of the arguments as the :meth:`~AddressParser.retrain` method. By default, all the parameters have
    the same default value as the :meth:`~AddressParser.retrain` method. The supported parameters are the following:

    - ``train_ratio``,
    - ``batch_size``,
    - ``epochs``,
    - ``num_workers``,
    - ``learning_rate``,
    - ``seed``,
    - ``logging_path``,
    - ``disable_tensorboard``,
    - ``layers_to_freeze``, and
    - ``name_of_the_retrain_parser``.


    Examples of usage:

    .. code-block:: sh

        retrain fasttext ./train_dataset_path.csv

    Using a gpu device

    .. code-block:: sh

        retrain bpemb ./train_dataset_path.csv --device 0

    Modifying training parameters

    .. code-block:: sh

        retrain bpemb ./train_dataset_path.csv --device 0 --batch_size 128 --learning_rate 0.001

    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]

    parsed_args = get_args(args)

    training_data = data_container_factory(
        dataset_path=parsed_args.train_dataset_path,
        trainable_dataset=True,
        csv_column_separator=parsed_args.csv_column_separator,
        csv_column_names=parsed_args.csv_column_names,
    )

    val_data = parsed_args.val_dataset_path
    if val_data is not None:
        val_data = data_container_factory(
            dataset_path=parsed_args.val_dataset_path,
            trainable_dataset=True,
            csv_column_separator=parsed_args.csv_column_separator,
            csv_column_names=parsed_args.csv_column_names,
        )

    base_parsing_model = parsed_args.base_parsing_model
    device = parsed_args.device

    if "cpu" not in device:
        device = int(device)
    parser_args = {"device": device, "cache_dir": parsed_args.cache_dir}
    parser_args_update_args = attention_model_type_handling(base_parsing_model)
    parser_args.update(**parser_args_update_args)

    address_parser = AddressParser(**parser_args)

    parsed_retain_arguments = parse_retrained_arguments(parsed_args)

    address_parser.retrain(
        train_dataset_container=training_data, val_dataset_container=val_data, **parsed_retain_arguments
    )


def get_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for the cli."""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    add_base_parsing_model_arg(parser)

    parser.add_argument(
        "train_dataset_path",
        help=wrap("The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format."),
        type=str,
    )

    parser.add_argument(
        "--val_dataset_path",
        help=wrap(
            "The path to the validation dataset file in a pickle (.p, .pickle or .pckl) or CSV format. "
            "If the dataset are CSV, both train and val must have the same CSV formatting "
            "(columns names). If not provided, the train dataset will be split in a train and val "
            "dataset (default is None)."
        ),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--train_ratio",
        help=wrap(
            "The ratio to use of the dataset for the training. The rest of the data is used for the "
            "validation (e.g. a training ratio of 0.8 mean an 80-20 train-valid split) (default is 0.8)."
        ),
        type=float,
        default=0.8,
    )

    add_batch_size_arg(parser)

    parser.add_argument(
        "--epochs",
        help=wrap("The number of training epochs (default is 5)."),
        type=int,
        default=5,
    )

    add_num_workers_arg(parser)

    parser.add_argument(
        "--learning_rate",
        help=wrap("The learning rate (LR) to use for training (default 0.01)."),
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--logging_path",
        help=wrap(
            "The logging path for the checkpoints and the retrained model. "
            "Note that training creates checkpoints, and we use Poutyne library that use the best epoch "
            "model and reloads the state if any checkpoints are already there. "
            "Thus, an error will be raised if you change the model type. For example, "
            "you retrain a FastText model and then retrain a BPEmb in the same logging path directory."
            "By default, the path is './checkpoints'."
        ),
        type=str,
        default="./checkpoints",
    )

    parser.add_argument(
        "--disable_tensorboard",
        help=wrap("To disable Poutyne automatic Tensorboard monitoring. By default, we disable them (true)."),
        type=bool_parse,
        default="True",
    )

    parser.add_argument(
        "--layers_to_freeze",
        help=wrap(
            "Name of the portion of the seq2seq to freeze layers, thus reducing the number of parameters to learn. "
            "Default to None."
        ),
        choices=[None, "encoder", "decoder", "prediction_layer", "seq2seq"],
        type=str,
        default=None,
    )

    parser.add_argument(
        "--name_of_the_retrain_parser",
        help=wrap(
            "Name to give to the retrained parser that will be used when reloaded as the printed name, "
            "and to the saving file name. By default, None, thus, the default name. See the complete parser retrain "
            "method for more details."
        ),
        default=None,
        type=str,
    )

    add_seed_arg(parser)

    add_device_arg(parser)

    add_csv_column_names_arg(parser)

    add_csv_column_separator_arg(parser)

    add_cache_dir_arg(parser)

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
