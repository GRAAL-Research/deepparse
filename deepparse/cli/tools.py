import pickle
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


def to_pickle(parsed_addresses: Union[FormattedParsedAddress, List[FormattedParsedAddress]], export_path: str) -> None:
    """
    Function to convert some parsed addresses into a list of list of tuples to be exported into a pickle file.
    """
    if isinstance(parsed_addresses, FormattedParsedAddress):
        parsed_addresses = [parsed_addresses]
    parsed_addresses = [parsed_address.to_pickle() for parsed_address in parsed_addresses]
    with open(export_path, "wb") as file:
        pickle.dump(parsed_addresses, file)
