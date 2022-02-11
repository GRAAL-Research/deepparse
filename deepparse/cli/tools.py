import os
import pickle
import textwrap
from typing import List, Union

import pandas as pd

from deepparse.parser import FormattedParsedAddress


def is_csv_path(dataset_path: str) -> bool:
    """
    Function to evaluate if a dataset path is a CSV file extension.

    Args:
        dataset_path (str): A dataset path.

    Return:
        Either or not, the path is a CSV file extension.
    """

    return ".csv" in dataset_path


def is_pickle_path(dataset_path: str) -> bool:
    """
    Function to evaluate if a dataset path is a pickle file extension.

    Args:
        dataset_path (str): A dataset path.

    Return:
        Either or not, the path is a pickle file extension.
    """
    return ".p" in dataset_path or ".pickle" in dataset_path


def to_csv(
    parsed_addresses: Union[FormattedParsedAddress, List[FormattedParsedAddress]], export_path: str, sep: str
) -> None:
    """
    Function to convert some parsed addresses into a dictionary to be exported into a CSV file using pandas.
    """
    if isinstance(parsed_addresses, FormattedParsedAddress):
        parsed_addresses = [parsed_addresses]
    csv_formatted_parsed_addresses = [parsed_address.to_pandas() for parsed_address in parsed_addresses]
    pd.DataFrame(csv_formatted_parsed_addresses).to_csv(export_path, sep=sep, index=False)
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


def generate_export_path(dataset_path: str, export_file_name: str) -> str:
    """
    Function to generate the export path by using the dataset path.

    Return:
        The string path.
    """
    return os.path.join(os.path.dirname(dataset_path), export_file_name)


# pylint: disable=pointless-string-statement
"""
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
    return '\n'.join(text)
