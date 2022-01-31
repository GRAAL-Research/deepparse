# pylint: disable=unbalanced-tuple-unpacking, W0102

import os
import pickle
import unittest
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

import pandas as pd

from deepparse import DataError
from deepparse.dataset_container import (
    PickleDatasetContainer,
    DatasetContainer,
    CSVDatasetContainer,
    comma_separated_list_reformat,
)

address_len = 6
base_string = "an address with the number {}"
a_tags_sequence = ["tag1", "tag2", "tag2", "tag3", "tag3", "tag4"]
default_csv_column_name = ["Address", "Tags"]


def create_data(number_of_data_points: int = 4) -> List:
    file_content = [
        (base_string.format(str(data_point)), a_tags_sequence) for data_point in range(number_of_data_points)
    ]

    return file_content


def create_pickle_file(path: str, number_of_data_points: int = 4) -> None:
    pickle_file_content = create_data(number_of_data_points)

    with open(path, "wb") as f:
        pickle.dump(pickle_file_content, f)


def create_csv_file(
    path: str,
    number_of_data_points: int = 4,
    column_names=default_csv_column_name,
    separator="\t",
    reformat_list_fn=None,
) -> None:
    csv_file_content = create_data(number_of_data_points)
    df = pd.DataFrame(csv_file_content, columns=column_names)
    if reformat_list_fn:
        df.Tags = df.Tags.apply(reformat_list_fn)
    df.to_csv(path, sep=separator, encoding="utf-8")


class ADatasetContainer(DatasetContainer):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.validate_dataset()


class DatasetContainerTest(TestCase):
    def tests_test_integration(self):
        some_valid_data = [("An address", [1, 0]), ("Another address", [2, 0]), ("A last address", [3, 4, 0])]
        a_dataset_container = ADatasetContainer(some_valid_data)
        self.assertIsNotNone(a_dataset_container.data)

    def test_when_not_list_of_tuple_then_raise_type_error(self):
        some_invalid_data = [1, 0]
        with self.assertRaises(TypeError):
            ADatasetContainer(some_invalid_data)

        some_invalid_data = "An address"
        with self.assertRaises(TypeError):
            ADatasetContainer(some_invalid_data)

        some_invalid_data = {"An address": [1, 0]}
        with self.assertRaises(TypeError):
            ADatasetContainer(some_invalid_data)

    def test_when_data_is_not_a_list_then_raise_type_error(self):
        some_invalid_data = ("An address", [1, 0])
        with self.assertRaises(TypeError):
            ADatasetContainer(some_invalid_data)

    def test_when_empty_address_then_raise_data_error(self):
        some_invalid_data = [("An address", [1, 0]), ("", []), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data)

    def test_when_whitespace_only_address_then_raise_data_error(self):
        some_invalid_data = [("An address", [1, 0]), (" ", []), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data)

        some_invalid_data = [("An address", [1, 0]), ("    ", []), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data)

    def test_when_empty_tags_set_then_raise_data_error(self):
        some_invalid_data = [("An address", [1, 0]), ("another address", []), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data)

    def test_when_tags_set_not_same_len_as_address_then_raise_data_error(self):
        some_invalid_data = [("An address", [1, 0]), ("another address", [1]), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data)

        some_invalid_data = [("An address", [1, 0]), ("another address", [1, 2, 4]), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data)


class PickleDatasetContainerTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()

        self.a_pickle_data_container_path = os.path.join(self.temp_dir_obj.name, "fake_pickle_data_container.p")

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_integration(self):
        number_of_data_points = 4
        create_pickle_file(
            self.a_pickle_data_container_path,
            number_of_data_points=number_of_data_points,
        )

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)
        expected = number_of_data_points
        self.assertEqual(expected, len(pickle_dataset_container))

        number_of_data_points = 5
        create_pickle_file(
            self.a_pickle_data_container_path,
            number_of_data_points=number_of_data_points,
        )

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)
        expected = number_of_data_points
        self.assertEqual(expected, len(pickle_dataset_container))

    def test_givenAPickleDatasetContainer_whenGetOneItem_thenReturnTheCorrectItem(self):
        create_pickle_file(self.a_pickle_data_container_path)

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)

        # first data point
        idx = 0
        expected_address = base_string.format(idx)
        expected_tags_idx = a_tags_sequence

        actual_address, actual_tags_idx = pickle_dataset_container[idx]
        self.assertEqual(expected_address, actual_address)
        self.assertListEqual(expected_tags_idx, actual_tags_idx)

        # second data point
        idx = 1
        expected_address = base_string.format(idx)
        expected_tags_idx = a_tags_sequence

        actual_address, actual_tags_idx = pickle_dataset_container[idx]
        self.assertEqual(expected_address, actual_address)
        self.assertListEqual(expected_tags_idx, actual_tags_idx)

        # third data point
        idx = 2
        expected_address = base_string.format(idx)
        expected_tags_idx = a_tags_sequence

        actual_address, actual_tags_idx = pickle_dataset_container[idx]
        self.assertEqual(expected_address, actual_address)
        self.assertListEqual(expected_tags_idx, actual_tags_idx)

    def test_givenAPickleDatasetContainer_whenGetSlice_thenReturnTheCorrectItems(self):
        create_pickle_file(self.a_pickle_data_container_path)

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)

        start_idx = 0
        end_idx = 2
        expected_addresses = [base_string.format(idx) for idx in range(start_idx, end_idx)]
        expected_tags_idxs = [a_tags_sequence] * (end_idx - start_idx)

        sliced_addresses = pickle_dataset_container[start_idx:end_idx]
        self.assertIsInstance(sliced_addresses, list)
        for actual_address_tuple, expected_address, expected_tags_idx in zip(
            sliced_addresses, expected_addresses, expected_tags_idxs
        ):
            actual_address, actual_tags_idx = actual_address_tuple[0], actual_address_tuple[1]
            self.assertEqual(expected_address, actual_address)
            self.assertListEqual(expected_tags_idx, actual_tags_idx)

        start_idx = 2
        end_idx = 4
        expected_addresses = [base_string.format(idx) for idx in range(start_idx, end_idx)]
        expected_tags_idxs = [a_tags_sequence] * (end_idx - start_idx)

        sliced_addresses = pickle_dataset_container[start_idx:end_idx]
        self.assertIsInstance(sliced_addresses, list)
        for actual_address_tuple, expected_address, expected_tags_idx in zip(
            sliced_addresses, expected_addresses, expected_tags_idxs
        ):
            actual_address, actual_tags_idx = actual_address_tuple[0], actual_address_tuple[1]
            self.assertEqual(expected_address, actual_address)
            self.assertListEqual(expected_tags_idx, actual_tags_idx)


class CSVDatasetContainerTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()

        self.a_data_container_path = os.path.join(self.temp_dir_obj.name, "fake_csv_data_container.csv")

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def _test_integration(self, number_of_data_points, csv_dataset_container):
        expected = number_of_data_points
        self.assertEqual(expected, len(csv_dataset_container))

    def test_integration(self):
        number_of_data_points = 4
        create_csv_file(
            self.a_data_container_path,
            number_of_data_points=number_of_data_points,
        )

        csv_dataset_container = CSVDatasetContainer(self.a_data_container_path, column_names=default_csv_column_name)
        self._test_integration(number_of_data_points, csv_dataset_container)

        number_of_data_points = 5
        create_csv_file(
            self.a_data_container_path,
            number_of_data_points=number_of_data_points,
        )

        csv_dataset_container = CSVDatasetContainer(self.a_data_container_path, column_names=default_csv_column_name)
        self._test_integration(number_of_data_points, csv_dataset_container)

    def test_integration_user_define_column_names(self):
        user_define_column_names = ["a_name", "Another_name"]
        number_of_data_points = 4
        create_csv_file(
            self.a_data_container_path,
            number_of_data_points=number_of_data_points,
            column_names=user_define_column_names,
        )

        csv_dataset_container = CSVDatasetContainer(self.a_data_container_path, column_names=user_define_column_names)

        self._test_integration(number_of_data_points, csv_dataset_container)

        number_of_data_points = 5
        create_csv_file(
            self.a_data_container_path,
            number_of_data_points=number_of_data_points,
            column_names=user_define_column_names,
        )

        csv_dataset_container = CSVDatasetContainer(self.a_data_container_path, column_names=user_define_column_names)

        self._test_integration(number_of_data_points, csv_dataset_container)

    def test_integration_user_define_separator(self):
        separator = ";"
        number_of_data_points = 4
        create_csv_file(self.a_data_container_path, number_of_data_points=number_of_data_points, separator=separator)

        csv_dataset_container = CSVDatasetContainer(
            self.a_data_container_path, column_names=default_csv_column_name, separator=separator
        )

        self._test_integration(number_of_data_points, csv_dataset_container)

        number_of_data_points = 5
        create_csv_file(self.a_data_container_path, number_of_data_points=number_of_data_points, separator=separator)

        csv_dataset_container = CSVDatasetContainer(
            self.a_data_container_path, column_names=default_csv_column_name, separator=separator
        )

        self._test_integration(number_of_data_points, csv_dataset_container)

    def test_integration_user_define_tag_separator_fn(self):
        def internal_conversion_of_list_to_comma_separated_list(tags):
            return str(tags).replace("[", "").replace("]", "").replace("'", "")

        number_of_data_points = 4
        create_csv_file(
            self.a_data_container_path,
            number_of_data_points=number_of_data_points,
            reformat_list_fn=internal_conversion_of_list_to_comma_separated_list,
        )

        csv_dataset_container = CSVDatasetContainer(
            self.a_data_container_path,
            column_names=default_csv_column_name,
            tag_seperator_reformat_fn=comma_separated_list_reformat,
        )

        self._test_integration(number_of_data_points, csv_dataset_container)

        number_of_data_points = 5
        create_csv_file(
            self.a_data_container_path,
            number_of_data_points=number_of_data_points,
            reformat_list_fn=internal_conversion_of_list_to_comma_separated_list,
        )

        csv_dataset_container = CSVDatasetContainer(
            self.a_data_container_path,
            column_names=default_csv_column_name,
            tag_seperator_reformat_fn=comma_separated_list_reformat,
        )

        self._test_integration(number_of_data_points, csv_dataset_container)

    def test_givenACSVDatasetContainer_whenGetOneItem_thenReturnTheCorrectItem(self):
        create_csv_file(self.a_data_container_path)

        csv_dataset_container = CSVDatasetContainer(self.a_data_container_path, column_names=default_csv_column_name)

        # first data point
        idx = 0
        expected_address = base_string.format(idx)
        expected_tags_idx = a_tags_sequence

        actual_address, actual_tags_idx = csv_dataset_container[idx]
        self.assertEqual(expected_address, actual_address)
        self.assertListEqual(expected_tags_idx, actual_tags_idx)

        # second data point
        idx = 1
        expected_address = base_string.format(idx)
        expected_tags_idx = a_tags_sequence

        actual_address, actual_tags_idx = csv_dataset_container[idx]
        self.assertEqual(expected_address, actual_address)
        self.assertListEqual(expected_tags_idx, actual_tags_idx)

        # third data point
        idx = 2
        expected_address = base_string.format(idx)
        expected_tags_idx = a_tags_sequence

        actual_address, actual_tags_idx = csv_dataset_container[idx]
        self.assertEqual(expected_address, actual_address)
        self.assertListEqual(expected_tags_idx, actual_tags_idx)

    def test_givenAPickleDatasetContainer_whenGetSlice_thenReturnTheCorrectItems(self):
        create_csv_file(self.a_data_container_path)

        csv_dataset_container = CSVDatasetContainer(self.a_data_container_path, column_names=default_csv_column_name)

        start_idx = 0
        end_idx = 2
        expected_addresses = [base_string.format(idx) for idx in range(start_idx, end_idx)]
        expected_tags_idxs = [a_tags_sequence] * (end_idx - start_idx)

        sliced_addresses = csv_dataset_container[start_idx:end_idx]
        self.assertIsInstance(sliced_addresses, list)
        for actual_address_tuple, expected_address, expected_tags_idx in zip(
            sliced_addresses, expected_addresses, expected_tags_idxs
        ):
            actual_address, actual_tags_idx = actual_address_tuple[0], actual_address_tuple[1]
            self.assertEqual(expected_address, actual_address)
            self.assertListEqual(expected_tags_idx, actual_tags_idx)

        start_idx = 2
        end_idx = 4
        expected_addresses = [base_string.format(idx) for idx in range(start_idx, end_idx)]
        expected_tags_idxs = [a_tags_sequence] * (end_idx - start_idx)

        sliced_addresses = csv_dataset_container[start_idx:end_idx]
        self.assertIsInstance(sliced_addresses, list)
        for actual_address_tuple, expected_address, expected_tags_idx in zip(
            sliced_addresses, expected_addresses, expected_tags_idxs
        ):
            actual_address, actual_tags_idx = actual_address_tuple[0], actual_address_tuple[1]
            self.assertEqual(expected_address, actual_address)
            self.assertListEqual(expected_tags_idx, actual_tags_idx)


if __name__ == "__main__":
    unittest.main()
