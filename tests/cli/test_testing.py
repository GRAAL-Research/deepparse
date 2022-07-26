# pylint: disable=too-many-arguments, too-many-locals

import logging
import os
import unittest
from unittest import skipIf
from unittest.mock import patch

import pytest
import torch

from deepparse.cli import test, generate_export_path
from tests.parser.base import PretrainedWeightsBase
from tests.parser.integration.base_retrain import RetrainTestCase


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class TestingTests(RetrainTestCase, PretrainedWeightsBase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    @classmethod
    def setUpClass(cls):
        super(TestingTests, cls).setUpClass()

        cls.download_pre_trained_weights(cls)

        cls.a_fasttext_model_type = "fasttext"
        cls.a_fasttext_att_model_type = "fasttext-attention"
        cls.a_fasttext_light_model_type = "fasttext-light"
        cls.a_bpemb_model_type = "bpemb"
        cls.a_bpemb_att_model_type = "bpemd-attention"

        cls.cpu_device = "cpu"
        cls.gpu_device = "0"

        cls.a_named_model = "a_retrained_model"

        cls.fasttext_parser_formatted_name = "PreTrainedFastTextAddressParser"
        cls.bpemb_parser_formatted_name = "PreTrainedBPEmbAddressParser"
        cls.fasttext_att_parser_formatted_name = "PreTrainedFastTextAttentionAddressParser"

        cls.a_cache_dir = "a_cache_dir"

    def test_integration_cpu(self):
        parser_params = [
            self.a_fasttext_model_type,
            self.a_train_pickle_dataset_path,
            "--device",
            self.cpu_device,
        ]

        test.main(parser_params)

        expected_file_path = generate_export_path(
            self.a_fasttext_model_type, f"{self.fasttext_parser_formatted_name}_testing.tsv"
        )

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir_obj.name, "data", expected_file_path)))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_integration_gpu(self):
        parser_params = [
            self.a_fasttext_model_type,
            self.a_train_pickle_dataset_path,
            "--device",
            self.gpu_device,
        ]

        test.main(parser_params)

        expected_file_path = generate_export_path(
            self.a_fasttext_model_type, f"{self.fasttext_parser_formatted_name}_testing.tsv"
        )

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir_obj.name, "data", expected_file_path)))

    def test_integration_logging(self):
        with self._caplog.at_level(logging.INFO):
            parser_params = [
                self.a_fasttext_model_type,
                self.a_train_pickle_dataset_path,
                "--device",
                self.cpu_device,
            ]

            test.main(parser_params)

        data_file_path = os.path.join(self.temp_dir_obj.name, 'data', self.a_train_pickle_dataset_path)

        expected_first_message = (
            f"Testing results on dataset file {data_file_path} using the parser {self.fasttext_parser_formatted_name}."
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

        expected_file_path = os.path.join(
            self.temp_dir_obj.name,
            "data",
            generate_export_path(self.a_fasttext_model_type, f"{self.fasttext_parser_formatted_name}_testing.tsv"),
        )
        expected_second_message = (
            f"Testing on the dataset file {data_file_path} is finished. "
            f"The results are logged in the CSV file at {expected_file_path}."
        )
        actual_second_message = self._caplog.records[1].message
        self.assertEqual(expected_second_message, actual_second_message)

    def test_integration_no_logging(self):
        with self._caplog.at_level(logging.INFO):
            parser_params = [
                self.a_fasttext_model_type,
                self.a_train_pickle_dataset_path,
                "--device",
                self.cpu_device,
                "--log",
                "False",
            ]

            test.main(parser_params)

        self.assertEqual(0, len(self._caplog.records))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_integration_attention_model(self):
        parser_params = [
            self.a_fasttext_att_model_type,
            self.a_train_pickle_dataset_path,
            "--device",
            self.gpu_device,
        ]

        test.main(parser_params)

        expected_file_path = generate_export_path(
            self.a_fasttext_att_model_type, f"{self.fasttext_att_parser_formatted_name}_testing.tsv"
        )

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir_obj.name, "data", expected_file_path)))

    def test_integration_csv(self):
        parser_params = [
            self.a_fasttext_model_type,
            self.a_train_csv_dataset_path,
            "--device",
            self.cpu_device,
            "--csv_column_names",
            "Address",
            "Tags",
            "--csv_column_separator",  # Our dataset use a comma as separator
            ",",
        ]

        test.main(parser_params)

        expected_file_path = generate_export_path(
            self.a_fasttext_model_type, f"{self.fasttext_parser_formatted_name}_testing.tsv"
        )

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir_obj.name, "data", expected_file_path)))

    def test_ifIsCSVFile_noColumnName_raiseValueError(self):
        with self.assertRaises(ValueError):
            parser_params = [
                self.a_fasttext_model_type,
                self.a_train_csv_dataset_path,
                "--device",
                self.cpu_device,
            ]

            test.main(parser_params)

    def test_ifIsNotSupportedFile_raiseValueError(self):
        with self.assertRaises(ValueError):
            parser_params = [
                self.a_fasttext_model_type,
                "an_unsupported_extension.json",
                "--device",
                self.cpu_device,
            ]

            test.main(parser_params)

    def test_ifPathToFakeRetrainModel_thenUseFakeRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            # We use the default path to fasttext model as a "retrain model path"
            path_to_retrained_model = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.ckpt")

            parser_params = [
                self.a_fasttext_model_type,
                self.a_train_pickle_dataset_path,
                "--path_to_retrained_model",
                path_to_retrained_model,
                "--device",
                self.cpu_device,
            ]

            test.main(parser_params)

        data_file_path = os.path.join(self.temp_dir_obj.name, 'data', self.a_train_pickle_dataset_path)

        expected_first_message = (
            f"Testing results on dataset file {data_file_path} using the parser {self.fasttext_parser_formatted_name}."
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

        expected_file_path = os.path.join(
            self.temp_dir_obj.name,
            "data",
            generate_export_path(self.a_fasttext_model_type, f"{self.fasttext_parser_formatted_name}_testing.tsv"),
        )
        expected_second_message = (
            f"Testing on the dataset file {data_file_path} is finished. "
            f"The results are logged in the CSV file at {expected_file_path}."
        )
        actual_second_message = self._caplog.records[1].message
        self.assertEqual(expected_second_message, actual_second_message)

    def test_ifPathToBPEmbRetrainModel_thenUseBPEmbRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            parser_params = [
                self.a_bpemb_model_type,
                self.a_train_pickle_dataset_path,
                "--device",
                self.cpu_device,
            ]

            test.main(parser_params)

        data_file_path = os.path.join(self.temp_dir_obj.name, 'data', self.a_train_pickle_dataset_path)

        expected_first_message = (
            f"Testing results on dataset file {data_file_path} using the parser {self.bpemb_parser_formatted_name}."
        )
        # Not the same position as with fasttext due to BPEmb messages
        actual_first_message = self._caplog.records[2].message
        self.assertEqual(expected_first_message, actual_first_message)

        expected_file_path = os.path.join(
            self.temp_dir_obj.name,
            "data",
            generate_export_path(self.a_fasttext_model_type, f"{self.bpemb_parser_formatted_name}_testing.tsv"),
        )
        expected_second_message = (
            f"Testing on the dataset file {data_file_path} is finished. "
            f"The results are logged in the CSV file at {expected_file_path}."
        )
        # Not the same position as with fasttext due to BPEmb messages
        actual_second_message = self._caplog.records[3].message
        self.assertEqual(expected_second_message, actual_second_message)

    def test_ifCachePath_thenUseNewCachePath(self):
        with patch("deepparse.cli.test.AddressParser") as address_parser_mock:
            parser_params = [
                self.a_fasttext_model_type,
                self.a_train_pickle_dataset_path,
                "--cache_dir",
                self.a_cache_dir,
                "--device",
                self.cpu_device,
            ]
            test.main(parser_params)

            address_parser_mock.assert_called()
            address_parser_mock.assert_called_with(
                device=self.cpu_device, cache_dir=self.a_cache_dir, model_type=self.a_fasttext_model_type
            )


if __name__ == "__main__":
    unittest.main()
