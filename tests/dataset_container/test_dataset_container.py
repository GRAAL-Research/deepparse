# pylint: disable=unbalanced-tuple-unpacking

import os
import unittest
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

from deepparse import DataError
from deepparse.dataset_container import (
    PickleDatasetContainer,
    DatasetContainer,
    CSVDatasetContainer,
)
from tests.tools import base_string, a_tags_sequence, create_pickle_file, create_csv_file, default_csv_column_name


def comma_separated_list_reformat(tags: str) -> List:
    """
    Function to parse a comma separated "list" of tag.

    Args:
        tags (str): A tag set string to parse.

    Return:
        A list of the parsed tag set.
    """
    return tags.split(", ")


class ADatasetContainer(DatasetContainer):
    def __init__(self, data, is_training_container=True):
        super().__init__(is_training_container)
        self.data = data
        self.validate_dataset()


class DatasetContainerTest(TestCase):
    def test_integration(self):
        some_valid_data = [("An address", [1, 0]), ("Another address", [2, 0]), ("A last address", [3, 4, 0])]
        a_dataset_container = ADatasetContainer(some_valid_data)
        self.assertIsNotNone(a_dataset_container.data)

    def test_integration_predict_container(self):
        some_valid_data = ["An address", "Another address", "A last address"]
        a_dataset_container = ADatasetContainer(some_valid_data, is_training_container=False)
        self.assertIsNotNone(a_dataset_container.data)

    def test_integration_slicing(self):
        some_valid_data = ["An address", "Another address", "A last address"]
        a_dataset_container = ADatasetContainer(some_valid_data, is_training_container=False)
        expected = 2
        self.assertEqual(len(a_dataset_container[0:2]), expected)

        some_valid_data = ["An address", "Another address", "A last address"]
        a_dataset_container = ADatasetContainer(some_valid_data, is_training_container=False)
        expected = 2
        self.assertEqual(len(a_dataset_container[:2]), expected)

        some_valid_data = ["An address", "Another address", "A last address"]
        a_dataset_container = ADatasetContainer(some_valid_data, is_training_container=False)
        expected = 1
        self.assertEqual(len(a_dataset_container[1:2]), expected)

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

    def test_when_none_then_raise_data_error(self):
        some_invalid_data = [("An address", [1, 0]), (None, []), ("A last address", [3, 4, 0])]
        with self.assertRaises(DataError):
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

    def test_when_predict_container_when_data_is_not_a_list_raise_type_error(self):
        some_invalid_data = "An address"
        with self.assertRaises(TypeError):
            ADatasetContainer(some_invalid_data, is_training_container=False)

    def test_when_predict_container_when_data_is_empty_raise_data_error(self):
        some_invalid_data = [""]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data, is_training_container=False)

    def test_when_predict_container_when_data_is_whitespace_only_raise_data_error(self):
        some_invalid_data = [" "]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data, is_training_container=False)

        some_invalid_data = ["    "]
        with self.assertRaises(DataError):
            ADatasetContainer(some_invalid_data, is_training_container=False)

    def test_when_training_container_when_is_data_set_container_return_true(self):
        some_valid_data = [("An address", [1, 0]), ("Another address", [2, 0]), ("A last address", [3, 4, 0])]
        a_dataset_container = ADatasetContainer(some_valid_data, is_training_container=True)
        self.assertTrue(a_dataset_container.is_training_container)

    def test_when_training_container_when_is_data_set_container_return_false(self):
        some_valid_data = ["An address", "Another address", "A last address"]
        a_dataset_container = ADatasetContainer(some_valid_data, is_training_container=False)
        self.assertFalse(a_dataset_container.is_training_container)


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

    def test_integration_predict_container(self):
        number_of_data_points = 4
        create_pickle_file(
            self.a_pickle_data_container_path, number_of_data_points=number_of_data_points, predict_container=True
        )

        pickle_dataset_container = PickleDatasetContainer(
            self.a_pickle_data_container_path, is_training_container=False
        )
        expected = number_of_data_points
        self.assertEqual(expected, len(pickle_dataset_container))

        number_of_data_points = 5
        create_pickle_file(
            self.a_pickle_data_container_path, number_of_data_points=number_of_data_points, predict_container=True
        )

        pickle_dataset_container = PickleDatasetContainer(
            self.a_pickle_data_container_path, is_training_container=False
        )
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

    def test_given_list_of_tuple_data_when_predict_container_raise_data_error(self):
        number_of_data_points = 4
        create_pickle_file(
            self.a_pickle_data_container_path, number_of_data_points=number_of_data_points, predict_container=False
        )

        with self.assertRaises(DataError):
            PickleDatasetContainer(self.a_pickle_data_container_path, is_training_container=False)


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

    def test_integration_predict_container(self):
        number_of_data_points = 4
        create_csv_file(
            self.a_data_container_path, number_of_data_points=number_of_data_points, predict_container=False
        )

        csv_dataset_container = CSVDatasetContainer(
            self.a_data_container_path, column_names=["Address"], is_training_container=False
        )
        self._test_integration(number_of_data_points, csv_dataset_container)

        number_of_data_points = 5
        create_csv_file(
            self.a_data_container_path, number_of_data_points=number_of_data_points, predict_container=False
        )

        csv_dataset_container = CSVDatasetContainer(
            self.a_data_container_path, column_names=["Address"], is_training_container=False
        )
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

    def test_given_a_training_container_when_column_names_not_2_raise_value_error(self):
        with self.assertRaises(ValueError):
            CSVDatasetContainer(self.a_data_container_path, column_names=[""])

        with self.assertRaises(ValueError):
            CSVDatasetContainer(self.a_data_container_path, column_names=["a single colum name"])

        with self.assertRaises(ValueError):
            CSVDatasetContainer(
                self.a_data_container_path, column_names=["a three colum name", "second name", "third name"]
            )

    def test_given_a_predict_container_when_column_names_not_1_raise_value_error(self):
        with self.assertRaises(ValueError):
            CSVDatasetContainer(
                self.a_data_container_path,
                column_names=["a two colum name", "second name"],
                is_training_container=False,
            )

        with self.assertRaises(ValueError):
            CSVDatasetContainer(
                self.a_data_container_path,
                column_names=["a three colum name", "second name", "third name"],
                is_training_container=False,
            )

    def test_given_a_data_container_with_improper_column_names_raise_value_error(self):
        with self.assertRaises(ValueError):
            CSVDatasetContainer(self.a_data_container_path, column_names=[""], is_training_container=False)

        with self.assertRaises(ValueError):
            CSVDatasetContainer(self.a_data_container_path, column_names=[" "], is_training_container=False)

        with self.assertRaises(ValueError):
            CSVDatasetContainer(self.a_data_container_path, column_names=["", ""], is_training_container=True)

        with self.assertRaises(ValueError):
            CSVDatasetContainer(self.a_data_container_path, column_names=[" ", " "], is_training_container=True)


if __name__ == "__main__":
    unittest.main()
