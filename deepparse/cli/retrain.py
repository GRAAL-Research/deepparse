import argparse
import sys
from typing import Dict

from deepparse.cli.tools import (
    is_csv_path,
    is_pickle_path,
    wrap,
    bool_parse,
    attention_model_type_handling,
)
from deepparse.dataset_container import CSVDatasetContainer, PickleDatasetContainer
from deepparse.parser import AddressParser

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
    CLI function to rapidly fine-tuned an addresses parser and saves it. One can retrain a base pretrained model
    using most of the arguments as the :meth:`~AddressParser.retrain` method. By default, all the parameters have
    the same default value as the :meth:`~AddressParser.retrain` method. The supported parameters are the following:

    - `train_ratio`,
    - `batch_size`,
    - `epochs`,
    - `num_workers`,
    - `learning_rate`,
    - `seed`,
    - `logging_path`,
    - `disable_tensorboard`,
    - `layers_to_freeze`, and
    - `name_of_the_retrain_parser`.


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

    train_dataset_path = parsed_args.train_dataset_path
    if is_csv_path(train_dataset_path):
        csv_column_names = parsed_args.csv_column_names
        if csv_column_names is None:
            raise ValueError(
                "To use a CSV dataset to retrain on, you need to specify the 'csv_column_names' argument to provide the"
                " column names to extract address and labels (respectively). For example, Address Tags."
            )
        csv_column_separator = parsed_args.csv_column_separator
        training_data = CSVDatasetContainer(
            train_dataset_path,
            column_names=csv_column_names,
            separator=csv_column_separator,
            is_training_container=True,
        )
    elif is_pickle_path(train_dataset_path):
        training_data = PickleDatasetContainer(train_dataset_path, is_training_container=True)
    else:
        raise ValueError("The train dataset path argument is not a CSV or a pickle file.")

    base_parsing_model = parsed_args.base_parsing_model
    device = parsed_args.device

    if "cpu" not in device:
        device = int(device)
    parser_args = {"device": device}
    parser_args_update_args = attention_model_type_handling(base_parsing_model)
    parser_args.update(**parser_args_update_args)

    address_parser = AddressParser(**parser_args)

    parsed_retain_arguments = parse_retrained_arguments(parsed_args)

    address_parser.retrain(dataset_container=training_data, **parsed_retain_arguments)


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
        help=wrap("The base parsing module to use for retraining."),
    )

    parser.add_argument(
        "train_dataset_path",
        help=wrap("The path to the dataset file in a pickle (.p, .pickle or .pckl) or CSV format."),
        type=str,
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

    parser.add_argument(
        "--batch_size",
        help=wrap("The size of the batch (default is 32)."),
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        help=wrap("The number of training epochs (default is 5)."),
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num_workers",
        help=wrap("The number of workers to use for the data loader (default is 1 worker)."),
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning_rate",
        help=wrap("The learning rate (LR) to use for training (default 0.01)."),
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        help=wrap("The seed to use (default 42)."),
        type=int,
        default=42,
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

    parser.add_argument(
        "--device",
        help=wrap("The device to use. It can be 'cpu' or a GPU device index such as '0' or '1'. By default, '0'."),
        type=str,
        default="0",
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
            " By default, '\t'."
        ),
        default="\t",
    )

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
