import os
import unittest
from tempfile import TemporaryDirectory
from unittest import skipIf

import torch

from deepparse.parser.tools import indices_splitting, load_tuple_to_device, validate_if_new_prediction_tags, \
    validate_if_new_seq2seq_params, get_address_parser_in_directory, get_files_in_directory, \
    pretrained_parser_in_directory
from tests.base_capture_output import CaptureOutputTestCase
from tests.tools import create_file


class ToolsTests(CaptureOutputTestCase):

    def setUp(self) -> None:
        self.a_seed = 42
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_directory = self.temp_dir_obj.name

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def populate_directory(self, with_retrain_parser: bool = False):
        os.makedirs(os.path.join(self.fake_directory, "a_directory"), exist_ok=True)
        create_file(os.path.join(self.fake_directory, "afile.txt"), "a content")
        create_file(os.path.join(self.fake_directory, "another_file.txt"), "a content")
        create_file(os.path.join(self.fake_directory, "random_file.txt"), "a content")

        checkpoints_dir_path = os.path.join(self.fake_directory, "checkpoints_dir")
        os.makedirs(checkpoints_dir_path, exist_ok=True)
        create_file(os.path.join(checkpoints_dir_path, "random_file.txt"), "a content")
        if with_retrain_parser:
            create_file(os.path.join(checkpoints_dir_path, "retrained_fasttext_address_parser.ckpt"), "a content")

    def test_givenACheckpointNewTags_whenValidateIfNewTags_thenReturnTrue(self):
        a_checkpoint_weights = {"some_weights": [1, 2, 2], "prediction_tags": {"a_tag": 1}}
        actual = validate_if_new_prediction_tags(a_checkpoint_weights)

        self.assertTrue(actual)

    def test_givenACheckpointNoNewTags_whenValidateIfNewTags_thenReturnFalse(self):
        a_checkpoint_weights = {"some_weights": [1, 2, 2]}
        actual = validate_if_new_prediction_tags(a_checkpoint_weights)

        self.assertFalse(actual)

    def test_givenACheckpointNewParams_whenValidateIfParams_thenReturnTrue(self):
        a_checkpoint_weights = {"some_weights": [1, 2, 2], "seq2seq_params": {"params": 1}}
        actual = validate_if_new_seq2seq_params(a_checkpoint_weights)

        self.assertTrue(actual)

    def test_givenACheckpointNoNewParams_whenValidateIfParams_thenReturnFalse(self):
        a_checkpoint_weights = {"some_weights": [1, 2, 2]}
        actual = validate_if_new_seq2seq_params(a_checkpoint_weights)

        self.assertFalse(actual)

    def test_givenADirectoryWithARetrainedModel_whenPretrainedParserInDirectory_thenReturnTrue(self):
        self.populate_directory(with_retrain_parser=True)
        actual = pretrained_parser_in_directory(self.fake_directory)

        self.assertTrue(actual)

    def test_givenADirectoryWithoutARetrainedModel_whenPretrainedParserInDirectory_thenReturnFalse(self):
        self.populate_directory(with_retrain_parser=False)
        actual = pretrained_parser_in_directory(self.fake_directory)

        self.assertFalse(actual)

    def test_givenADirectory_whenGetFilesInDirectory_thenReturnListWithAllFiles(self):
        self.populate_directory()
        actual = get_files_in_directory(self.fake_directory)

        expected = ['afile.txt', 'random_file.txt', 'another_file.txt', 'random_file.txt']

        for actual_element in actual:
            self.assertIn(actual_element, expected)
        self.assertEqual(len(actual), len(expected))

    def test_givenAEmptyDirectory_whenGetFilesInDirectory_thenReturnEmptyList(self):
        actual = get_files_in_directory(self.fake_directory)

        expected = []

        self.assertEqual(actual, expected)

    def test_givenAEmptyDirectory_whenGetAddressParserInDirectory_thenReturnEmptyList(self):
        a_list_of_directory = ["afile.txt", "another_file.txt"]
        actual = get_address_parser_in_directory(a_list_of_directory)

        expected = []

        self.assertEqual(actual, expected)

    def test_givenADirectoryWithARetrainParser_whenGetAddressParserInDirectory_thenReturnThePath(self):
        a_list_of_directory = ["afile.txt", "another_file.txt", "retrained_fasttext_address_parser.ckpt"]
        actual = get_address_parser_in_directory(a_list_of_directory)

        expected = ["retrained_fasttext_address_parser.ckpt"]

        self.assertEqual(actual, expected)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenATupleToLoadOfTensorIntoDevice_whenLoad_thenProperlyLoad(self):
        a_device = torch.device("cuda:0")
        a_random_tensor = torch.rand(1, 2)
        a_tuple_of_tensor = (a_random_tensor, a_random_tensor, a_random_tensor)

        actual = load_tuple_to_device(a_tuple_of_tensor, a_device)

        for element in actual:
            self.assertEqual(element.device, a_device)

    def test_givenADataset_whenIndicesSplittingRatio8020_thenSplitIndices80Train20Valid(self):
        number_of_data_points_in_dataset = 100
        train_ratio = 0.8
        expected_train_indices = [
            83, 53, 70, 45, 44, 39, 22, 80, 10, 0, 18, 30, 73, 33, 90, 4, 76, 77, 12, 31, 55, 88, 26, 42, 69, 15, 40,
            96, 9, 72, 11, 47, 85, 28, 93, 5, 66, 65, 35, 16, 49, 34, 7, 95, 27, 19, 81, 25, 62, 13, 24, 3, 17, 38, 8,
            78, 6, 64, 36, 89, 56, 99, 54, 43, 50, 67, 46, 68, 61, 97, 79, 41, 58, 48, 98, 57, 75, 32, 94, 59
        ]
        expected_valid_indices = [63, 84, 37, 29, 1, 52, 21, 2, 23, 87, 91, 74, 86, 82, 20, 60, 71, 14, 92, 51]
        expected_len_train_indices = 80
        expected_len_valid_indices = 20

        actual_train_indices, actual_valid_indices = indices_splitting(number_of_data_points_in_dataset,
                                                                       train_ratio,
                                                                       seed=self.a_seed)
        self.assertEqual(len(actual_train_indices), expected_len_train_indices)
        self.assertEqual(len(actual_valid_indices), expected_len_valid_indices)
        self.assertEqual(actual_train_indices, expected_train_indices)
        self.assertEqual(actual_valid_indices, expected_valid_indices)


if __name__ == "__main__":
    unittest.main()
