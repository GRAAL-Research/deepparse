# pylint: disable=too-many-arguments
from abc import ABC, abstractmethod
from pickle import load
from typing import Union, List, Dict, Callable

import pandas as pd
from torch.utils.data import Dataset

from .tools import former_python_list, validate_column_names
from ..data_error import DataError
from ..data_validation import validate_if_any_empty, validate_if_any_whitespace_only, validate_if_any_none


class DatasetContainer(Dataset, ABC):
    """
    Interface for the dataset. This interface defines most of the methods that the dataset needs to define.
    If you define another dataset container, the init must define the attribute data.

    We also recommend using the ``validate_dataset`` method in your init to validate some characteristics of your
    dataset.

    For a training container, it validates the following:

        - all addresses are not None value,
        - all addresses are not empty,
        - all addresses are not whitespace string,
        - all tags are not empty, if data is a list of tuple (``[('an address', ['a_tag', 'another_tag']), ...]``), and
        - if the addresses (whitespace-split) are the same length as their respective tags list.

    While for a predict container (unknown prediction tag), it validates the following:

        - all addresses are not None,
        - all addresses are not empty, and
        - all addresses are not whitespace string.

    Args:
        is_training_container (bool): Either or not, the dataset container is a training container. This will determine
            the dataset validation test we apply to the dataset. That is, a predict dataset doesn't include tags.
            The default value is true.
    """

    @abstractmethod
    def __init__(self, is_training_container: bool = True) -> None:
        """
        Need to be defined by the child class.
        """
        self.data = None
        self.is_training_container = is_training_container

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            start, stop, _ = idx.indices(len(self))
            result = [self.data[index] for index in range(start, stop)]
        else:
            result = self.data[idx]
        return result

    def validate_dataset(self) -> None:
        if not self._data_is_a_list():
            raise TypeError("The dataset is not a list.")

        if self.is_training_container:
            data_to_validate = [data[0] for data in self.data]
        else:
            data_to_validate = self.data

        if validate_if_any_none(string_elements=data_to_validate):
            raise DataError("Some addresses data points are None value.")

        if self.is_training_container:
            # Not done in previous similar if since none test not applied
            self._training_validation()

        if validate_if_any_empty(string_elements=data_to_validate):
            raise DataError("Some addresses data points are empty.")

        if validate_if_any_whitespace_only(string_elements=data_to_validate):
            raise DataError("Some addresses only include whitespace thus cannot be parsed.")

    def _data_is_a_list(self):
        return isinstance(self.data, list)

    def _training_validation(self):
        if not self._data_is_list_of_tuple():
            raise TypeError(
                "The pickled dataset data are not in a tuple format. Data"
                "is expected to be a list of tuples where the first element is"
                "the address, and the second is the address tag."
            )

        if self._empty_tags():
            raise DataError("Some tags data points are empty.")

        if not self._data_tags_is_same_len_then_address():
            raise DataError("Some addresses (whitespace-split) and the tags associated with them are not the same len.")

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

    def is_a_train_container(self) -> bool:
        return self.is_training_container


class PickleDatasetContainer(DatasetContainer):
    """
    Pickle dataset container that imports a list of addresses in pickle format and does some validation on it.

    The dataset needs to be a list of tuples where the first element of each tuple is the address (a string),
    and the second is a list of the expected tag to predict (e.g. ``[('an address', ['a_tag', 'another_tag']), ...]``).
    The len of the tags needs to be the same as the len of the address when whitespace split.

    For a training container, the validation tests applied on the dataset are the following:

        - all addresses are not None value,
        - all addresses are not empty,
        - all addresses are not whitespace string,
        - all tags are not empty, if data is a list of tuple (``[('an address', ['a_tag', 'another_tag']), ...]``), and
        - if the addresses (whitespace-split) are the same length as their respective tags list.

    While for a predict container (unknown prediction tag), the validation tests applied on the dataset are the
    following:

        - all addresses are not None value,
        - all addresses are not empty, and
        - all addresses are not whitespace string.

    Args:
        data_path (str): The path to the pickle dataset file.
        is_training_container (bool): Either or not, the dataset container is a training container. This will determine
            the dataset validation test we apply to the dataset. That is, a predict dataset doesn't include tags.
            The default value is true.

    """

    def __init__(self, data_path: str, is_training_container: bool = True) -> None:
        super().__init__(is_training_container=is_training_container)
        with open(data_path, "rb") as f:
            self.data = load(f)

        if not is_training_container:
            if self._test_predict_container_is_list_of_tuple():
                raise DataError(
                    "The data is a list of tuple by the dataset container is a predict container. "
                    "Predict container should contains only a list of address."
                )

        self.validate_dataset()

    def _test_predict_container_is_list_of_tuple(self) -> bool:
        return any((isinstance(data, tuple) for data in self.data))


class CSVDatasetContainer(DatasetContainer):
    """
    CSV dataset container that imports a CSV of addresses. If the dataset is a predict one, it needs to have at least
    one column with some addresses. If the dataset is a training one (with prediction tags), it needs to have at
    least two columns, one with some addresses and another with a list of tags for each address.

    After loading the CSV dataset, some tests will be applied depending on its type.

    For a training container, the validation tests applied on the dataset are the following:

        - all addresses are not None value,
        - all addresses are not empty,
        - all addresses are not whitespace string, and
        - if the addresses (whitespace-split) are the same length as their respective tags list.

    While for a predict container (unknown prediction tag), the validation tests applied on the dataset are the
    following:

        - all addresses are not None value,
        - all addresses are not empty, and
        - all addresses are not whitespace string.

    Args:

        data_path (str): The path to the CSV dataset file.
        column_names (List): A column name list to extract the dataset element.
            If the dataset container is a predict one, the list must be of exactly one element
            (i.e. the address column). On the other hand, if the dataset container is a training one, the list must be
            of exactly two elements: addresses and tags.
        is_training_container (bool): Either or not, the dataset container is a training container. This will determine
            the dataset validation test we apply to the dataset. That is, a predict dataset doesn't include tags.
            The default value is true.
        separator (str): The CSV columns separator to use. By default, ``"\\t"``.
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
        column_names: Union[List, str],
        is_training_container: bool = True,
        separator: str = "\t",
        tag_seperator_reformat_fn: Union[None, Callable] = None,
        csv_reader_kwargs: Union[None, Dict] = None,
    ) -> None:
        super().__init__(is_training_container=is_training_container)
        if is_training_container:
            if len(column_names) != 2:
                raise ValueError("When the dataset is a training container, two column names must be provided.")
        else:  # A predict container
            if len(list(column_names)) != 1:
                raise ValueError("When the dataset is a predict container, one column name must be provided.")

        if validate_column_names(column_names):
            raise ValueError("A column name is an empty string or whitespace only.")

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

        if is_training_container:
            data = [
                (data_point[0], tag_seperator_reformat_fn(data_point[1]))
                for data_point in pd.read_csv(**csv_reader_kwargs)[column_names].to_numpy()
            ]
        else:
            data = [data_point[0] for data_point in pd.read_csv(**csv_reader_kwargs)[column_names].to_numpy()]
        self.data = data
        self.validate_dataset()
