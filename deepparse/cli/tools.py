import argparse
import json
import os
import pickle
import textwrap
from typing import List, Union, Dict

import pandas as pd

from ..dataset_container import DatasetContainer, CSVDatasetContainer, PickleDatasetContainer
from ..parser import FormattedParsedAddress


def is_csv_path(export_file_name: str) -> bool:
    """
    Function to evaluate if a dataset path is a CSV file extension.

    Args:
        export_file_name (str): A export file name.

    Return:
        Either or not, the path is a CSV file extension.
    """

    return ".csv" in export_file_name


def is_pickle_path(export_file_name: str) -> bool:
    """
    Function to evaluate if a dataset path is a pickle file extension.

    Args:
        export_file_name (str): A export file name.

    Return:
        Either or not, the path is a pickle file extension.
    """
    return ".p" in export_file_name or ".pickle" in export_file_name or ".pckl" in export_file_name


def is_json_path(export_file_name: str) -> bool:
    """
    Function to evaluate if a dataset path is a json file extension.

    Args:
        export_file_name (str): A export file name.

    Return:
        Either or not, the path is a json file extension.
    """
    return ".json" in export_file_name


def to_csv(
    parsed_addresses: Union[FormattedParsedAddress, List[FormattedParsedAddress]], export_path: str, sep: str
) -> None:
    """
    Function to convert some parsed addresses into a dictionary to be exported into a CSV file using pandas.
    """
    if isinstance(parsed_addresses, FormattedParsedAddress):
        parsed_addresses = [parsed_addresses]
    nested_dict_formatted_parsed_addresses = [parsed_address.to_pandas() for parsed_address in parsed_addresses]
    pd.DataFrame(nested_dict_formatted_parsed_addresses).to_csv(export_path, sep=sep, index=False)
    print(f"Data exported to {export_path}.")


def to_pickle(parsed_addresses: Union[FormattedParsedAddress, List[FormattedParsedAddress]], export_path: str) -> None:
    """
    Function to convert some parsed addresses into a list of list of tuples to be exported into a pickle file.
    """
    if isinstance(parsed_addresses, FormattedParsedAddress):
        parsed_addresses = [parsed_addresses]
    parsed_addresses = [parsed_address.to_pickle() for parsed_address in parsed_addresses]
    with open(export_path, "wb") as file:
        pickle.dump(parsed_addresses, file)
    print(f"Data exported to {export_path}.")


def to_json(parsed_addresses: Union[FormattedParsedAddress, List[FormattedParsedAddress]], export_path: str) -> None:
    """
    Function to convert some parsed addresses into a json to be exported into a JSON file.
    """
    if isinstance(parsed_addresses, FormattedParsedAddress):
        parsed_addresses = [parsed_addresses]
    nested_dict_formatted_parsed_addresses = [parsed_address.to_pandas() for parsed_address in parsed_addresses]
    with open(export_path, "w", encoding="utf-8") as file:
        json.dump(nested_dict_formatted_parsed_addresses, file, ensure_ascii=False)
    print(f"Data exported to {export_path}.")


def bool_parse(arg: str) -> bool:
    """
    Function to fix the argparse bool parsing.

    Args:
        arg (str): The argument to bool parse.

    Return:
        A Python bool.
    """
    if arg.lower() in ("true", "t", "yes", "y", "1"):
        parsed_bool = True
    elif arg.lower() in ("false", "f", "no", "n", "0"):
        parsed_bool = False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    return parsed_bool


def generate_export_path(dataset_path: str, export_file_name: str) -> str:
    """
    Function to generate the export path by using the dataset path.

    Args:
        dataset_path (str): The dataset path to base the generated export path.
        export_file_name (str): The export file name to use for the path.

    Return:
        The string path.
    """
    return os.path.join(os.path.dirname(dataset_path), export_file_name)


def replace_path_extension(path: str, extension: str) -> str:
    """
    Function to replace a path extension.

    Args:
        path (str): The path to replace the extension.
        extension (str): The extension to use for the replacement.

    Return:
        The modified path using the new extension.
    """
    pre, _ = os.path.splitext(path)
    return os.path.join(pre + extension)


def attention_model_type_handling(parsing_model) -> Dict:
    """
    Function to handle the attention_mechanism flag base on model type name.
    Return:
        A params dict.
    """
    parser_args_update_args = {}
    if "-attention" in parsing_model:
        parser_args_update_args.update({"attention_mechanism": True})
        parsing_model = parsing_model.strip("attention").strip("-")
    parser_args_update_args.update({"model_type": parsing_model})
    return parser_args_update_args


def data_container_factory(
    dataset_path: str,
    trainable_dataset: bool,
    csv_column_separator: str = None,
    csv_column_name: str = None,
    csv_column_names: List = None,
) -> DatasetContainer:
    """
    Factory to create the trainable dataset container.
    """
    if is_csv_path(dataset_path):
        if trainable_dataset:
            # Train or test dataset case
            if csv_column_names is None:
                raise ValueError(
                    "To use a CSV dataset to retrain on, you need to specify the 'csv_column_names' argument to "
                    "provide the column names to extract address and labels (respectively). For example, Address Tags."
                )
        else:
            # Parse dataset
            if csv_column_name is None:
                raise ValueError(
                    "For a CSV dataset path, you need to specify the 'csv_column_name' argument to provide the"
                    " column name to extract address."
                )
            csv_column_names = csv_column_name
        data_container = CSVDatasetContainer(
            dataset_path,
            column_names=csv_column_names,
            separator=csv_column_separator,
            is_training_container=trainable_dataset,
        )
    elif is_pickle_path(dataset_path):
        data_container = PickleDatasetContainer(dataset_path, is_training_container=trainable_dataset)
    else:
        raise ValueError("The train dataset path argument is not a CSV or a pickle file.")

    return data_container


# pylint: disable=pointless-string-statement
PYPYR_COPYRIGHT_APACHE_LICENSE = """
The code below was copied from the pypyr project, and has been modified for the purpose of this package.

COPYRIGHT

All contributions from the https://github.com/pypyr/pypyr authors.
Copyright (c) 2018 - 2022
All rights reserved.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

LICENSE

The Apache License 2.0

See project for complete license.
"""


def wrap(text, **kwargs):  # pragma: no cover
    """Wrap lines in argparse, so they align nicely in 2 columns.
    Default width is 70.
    With gratitude to paul.j3 https://bugs.python.org/issue12806
    """
    # apply textwrap to each line individually
    text = text.splitlines()
    text = [textwrap.fill(line, **kwargs) for line in text]
    return "\n".join(text)
