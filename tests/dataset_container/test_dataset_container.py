# we disable the pylint no-member since he think the dataset container value are list
# pylint: disable=no-member

import os
import pickle
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from deepparse.dataset_container import DatasetContainer, PickleDatasetContainer


class ADatasetContainer(DatasetContainer):

    def __init__(self, number_of_data_points: int = 4, number_of_features_each: int = 10):
        super().__init__()
        self.data = np.arange(0, number_of_data_points * number_of_features_each).reshape(
            (number_of_data_points, number_of_features_each))


def create_pickle_file(path: str, number_of_data_points: int = 4, number_of_features_each: int = 10):
    pickle_file_content = np.arange(0, number_of_data_points * number_of_features_each).reshape(
        (number_of_data_points, number_of_features_each))

    with open(path, "wb") as f:
        pickle.dump(pickle_file_content, f)


class DataSetContainerTest(TestCase):

    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.a_pickle_data_container_path = os.path.join(self.temp_dir_obj.name, 'fake_pickle_data_container.p')

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def assertListOfArraysEqual(self, expected_list_arrays, actual_list_arrays):
        self.assertEqual(len(expected_list_arrays), len(actual_list_arrays))
        for expected_array, actual_array in zip(expected_list_arrays, actual_list_arrays):
            self.assertEqual(expected_array.tolist(), actual_array.tolist())

    def test_givenAPickleDatasetContainer_whenInstantiate_thenDataIsPickleContent(self):
        number_of_data_points = 4
        create_pickle_file(self.a_pickle_data_container_path, number_of_data_points=number_of_data_points)

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)
        expected = number_of_data_points
        self.assertEqual(expected, len(pickle_dataset_container))

        number_of_data_points = 5
        create_pickle_file(self.a_pickle_data_container_path, number_of_data_points=number_of_data_points)

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)
        expected = number_of_data_points
        self.assertEqual(expected, len(pickle_dataset_container))

    def test_givenAPickleDatasetContainer_whenGetOneItem_thenReturnTheCorrectItem(self):
        create_pickle_file(self.a_pickle_data_container_path)

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)

        expected = list(range(0, 10))  # first data point
        actual = pickle_dataset_container[0]
        self.assertEqual(expected, actual.tolist())

        expected = list(range(10, 20))  # second data point
        actual = pickle_dataset_container[1]
        self.assertEqual(expected, actual.tolist())

        expected = list(range(20, 30))  # third data point
        actual = pickle_dataset_container[2]
        self.assertEqual(expected, actual.tolist())

    def test_givenAPickleDatasetContainer_whenGetSlice_thenReturnTheCorrectItems(self):
        create_pickle_file(self.a_pickle_data_container_path)

        pickle_dataset_container = PickleDatasetContainer(self.a_pickle_data_container_path)

        expected = [np.array(range(0, 10)), np.array(range(10, 20))]  # first and second data points
        actual = pickle_dataset_container[0:2]
        self.assertListOfArraysEqual(expected, actual)

        expected = [np.array(range(20, 30)), np.array(range(30, 40))]  # third and forth data points
        actual = pickle_dataset_container[2:4]
        self.assertListOfArraysEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
