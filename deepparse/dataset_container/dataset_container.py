# pylint: disable=too-many-arguments
from abc import ABC, abstractmethod
from pickle import load
from typing import Union, List, Dict, Callable

import pandas as pd
from torch.utils.data import Dataset

from .tools import former_python_list
from ..data_error import DataError
from ..data_validation import is_whitespace_only_address, is_empty_address


class DatasetContainer(Dataset, ABC):
    """
    Interface for the dataset. This interface defines most of the methods that the dataset needs to define.
    If you define another dataset container, the init must define the attribute data.

    We also recommend using the ``validate_dataset`` method in your init to validate some characteristics of your
    dataset. It validates the following:

        - all addresses are not empty,
        - all addresses are not whitespace string,
        - all tags are not empty, if data is a list of tuple (``[('an address', ['a_tag', 'another_tag']), ...]``), and
        - if the addresses (whitespace-split) are the same length as their respective tags list.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Need to be defined by the child class.
        """
        self.data = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            result = []
            for element in range(idx.start, idx.stop):
                sample = self.data[element]

                result.append(sample)
        else:
            result = self.data[idx]

        return result

    def validate_dataset(self) -> None:
        if not self._data_is_a_list():
            raise TypeError("The dataset is not a list.")

        if not self._data_is_list_of_tuple():
            raise TypeError(
                "The pickled dataset data are not in a tuple format. Data"
                "is expected to be a list of tuples where the first element is"
                "the address, and the second is the address tag."
            )
        if self._empty_address():
            raise DataError("Some addresses data points are empty.")

        if self._whitespace_only_addresses():
            raise DataError("Some addresses only include whitespace thus cannot be parsed.")

        if self._empty_tags():
            raise DataError("Some tags data points are empty.")

        if not self._data_tags_is_same_len_then_address():
            raise DataError("Some addresses (whitespace-split) and the tags associated with them are not the same len.")

    def _data_is_a_list(self):
        return isinstance(self.data, list)

    def _empty_address(self) -> bool:
        """
        Return true if one of the addresses is an empty string.
        """
        return any((is_empty_address(data[0]) for data in self.data))

    def _whitespace_only_addresses(self) -> bool:
        """
        Return true if one the address is composed of only whitespace.
        """
        return any((is_whitespace_only_address(data[0]) for data in self.data))

    def _data_is_list_of_tuple(self) -> bool:
        """
        Return true if one of the elements in the dataset is not a tuple.
        """
        return all(isinstance(data, tuple) for data in self.data)

    def _empty_tags(self) -> bool:
        """
        Return true if one of the tag sets is empty.
        """
        return all(len(data[1]) == 0 for data in self.data)

    def _data_tags_is_same_len_then_address(self) -> bool:
        """
        Return true if all the data tags are the same len as the address split at each whitespace.
        """
        return all(len(data[0].split(" ")) == len(data[1]) for data in self.data)


class PickleDatasetContainer(DatasetContainer):
    """
    Pickle dataset container that imports a list of addresses in pickle format and does some validation on it.

    The dataset needs to be a list of tuples where the first element of each tuple is the address (a string),
    and the second is a list of the expected tag to predict (e.g. ``[('an address', ['a_tag', 'another_tag']), ...]``).
    The len of the tags needs to be the same as the len of the address when whitespace split. The validation tests
    applied on the dataset is the following:

        - all addresses are not empty,
        - all addresses are not whitespace string,
        - all tags are not empty, if data is a list of tuple (``[('an address', ['a_tag', 'another_tag']), ...]``), and
        - if the addresses (whitespace-split) are the same length as their respective tags list.

    Args:
        data_path (str): The path to the pickle dataset file.

    """

    def __init__(self, data_path: str) -> None:
        super().__init__()
        with open(data_path, "rb") as f:
            self.data = load(f)

        self.validate_dataset()


class CSVDatasetContainer(DatasetContainer):
    """
    CSV dataset container that imports a (at least) two columns CSV of addresses with one column as addresses data and
    another column as tag set per address.

    Args:

        data_path (str): The path to the CSV dataset file.
        column_names (List): A list of the names of the dataframe column to extract the addresses and tags,
            respectively.
        separator (str): The csv columns separator to use. By default, ``"\\t"``.
        tag_seperator_reformat_fn (Callable, optional): A function to parse a tags string and return a list of
            address tags. For example, if the tag column is a former python list saved with pandas, the characters ``]``
            , ``]`` and ``'`` will be included as the tags' element. Thus, a parsing function will take a string as is
            parameter and output a python list. The default function process it as a former python list.
            That is, it removes the ``[],`` characters and splits the sequence at each comma (``","``).
        csv_reader_kwargs (dict, optional): Keyword arguments to pass to pandas ``read_csv`` use internally. By default,
            the ``data_path`` is passed along with our default ``sep`` value ( ``"\\t"``) and the ``"utf-8"`` encoding
            format. However, this can be overridden by using this argument again.
    """

    def __init__(
        self,
        data_path: str,
        column_names: List,
        separator: str = "\t",
        tag_seperator_reformat_fn: Union[None, Callable] = None,
        csv_reader_kwargs: Union[None, Dict] = None,
    ) -> None:
        super().__init__()
        if csv_reader_kwargs is None:
            csv_reader_kwargs = {}
        csv_reader_kwargs = {
            "filepath_or_buffer": data_path,
            "sep": separator,
            "encoding": "utf-8",
            **csv_reader_kwargs,
        }
        if tag_seperator_reformat_fn is None:
            tag_seperator_reformat_fn = former_python_list
        self.data = [
            (data_point[0], tag_seperator_reformat_fn(data_point[1]))
            for data_point in pd.read_csv(**csv_reader_kwargs)[column_names].to_numpy()
        ]

        self.validate_dataset()
