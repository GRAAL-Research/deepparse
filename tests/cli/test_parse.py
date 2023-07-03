# pylint: disable=too-many-arguments, too-many-locals

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import logging
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import skipIf
from unittest.mock import patch

import pytest

from deepparse.cli import parse, generate_export_path
from tests.parser.base import PretrainedWeightsBase
from tests.tools import create_pickle_file, create_csv_file


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run for unit tests since download is too long.")
class ParseTests(PretrainedWeightsBase):
    @classmethod
    def setUpClass(cls):
        super(ParseTests, cls).setUpClass()
        cls.prepare_pre_trained_weights()

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()

        self.fake_data_path_pickle = os.path.join(self.temp_dir_obj.name, "fake_data.p")

        self.fake_data_path_csv = os.path.join(self.temp_dir_obj.name, "fake_data.csv")
        self.a_unsupported_data_path = os.path.join(self.temp_dir_obj.name, "fake_data.txt")

        self.pickle_p_export_filename = "a_file.p"
        self.pickle_pickle_export_filename = "a_file.pickle"
        self.csv_export_filename = "a_file.csv"
        self.json_export_filename = "a_file.json"

        self.a_fasttext_model_type = "fasttext"
        self.a_fasttext_att_model_type = "fasttext-attention"
        self.a_fasttext_light_model_type = "fasttext-light"
        self.a_bpemb_model_type = "bpemb"
        self.a_bpemb_att_model_type = "bpemd-attention"

        self.cpu_device = "cpu"
        self.gpu_device = "0"

        self.a_cache_dir = "a_cache_dir"

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_integration_cpu(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_model_type,
                self.fake_data_path_pickle,
                self.pickle_p_export_filename,
                "--device",
                self.cpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_filename)
        self.assertTrue(os.path.isfile(export_path))

    def test_integration_gpu(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_model_type,
                self.fake_data_path_pickle,
                self.pickle_p_export_filename,
                "--device",
                self.gpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_filename)
        self.assertTrue(os.path.isfile(export_path))

    def test_integration_logging(self):
        with self._caplog.at_level(logging.INFO):
            create_pickle_file(self.fake_data_path_pickle, predict_container=True)
            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.fake_data_path_pickle,
                    self.pickle_p_export_filename,
                    "--device",
                    self.cpu_device,
                ]
            )
        expected_first_message = (
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"PreTrainedFastTextAddressParser"
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

        export_path = generate_export_path(self.fake_data_path_pickle, "a_file.p")
        expected_second_message = (
            f"4 addresses have been parsed.\n" f"The parsed addresses are outputted here: {export_path}"
        )
        actual_second_message = self._caplog.records[1].message
        self.assertEqual(expected_second_message, actual_second_message)

    def test_integration_no_logging(self):
        with self._caplog.at_level(logging.INFO):
            create_pickle_file(self.fake_data_path_pickle, predict_container=True)
            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.fake_data_path_pickle,
                    self.pickle_p_export_filename,
                    "--device",
                    self.cpu_device,
                    "--log",
                    "False",
                ]
            )
        self.assertEqual(0, len(self._caplog.records))

    def test_integration_attention_model(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_att_model_type,
                self.fake_data_path_pickle,
                self.pickle_p_export_filename,
                "--device",
                self.cpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_filename)
        self.assertTrue(os.path.isfile(export_path))

    def test_integration_json(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_att_model_type,
                self.fake_data_path_pickle,
                self.json_export_filename,
                "--device",
                self.cpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.json_export_filename)
        self.assertTrue(os.path.isfile(export_path))

    def test_integration_csv(self):
        create_csv_file(self.fake_data_path_csv, predict_container=True)

        parse.main(
            [
                self.a_fasttext_att_model_type,
                self.fake_data_path_csv,
                self.csv_export_filename,
                "--device",
                self.cpu_device,
                "--csv_column_name",
                "Address",
            ]
        )

        export_path = generate_export_path(self.fake_data_path_csv, self.csv_export_filename)
        self.assertTrue(os.path.isfile(export_path))

    def test_integration_csv_separator(self):
        sep = ";"
        create_csv_file(self.fake_data_path_csv, predict_container=True, separator=sep)

        parse.main(
            [
                self.a_fasttext_model_type,
                self.fake_data_path_csv,
                self.csv_export_filename,
                "--device",
                self.cpu_device,
                "--csv_column_name",
                "Address",
                "--csv_column_separator",
                sep,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.csv_export_filename)
        self.assertTrue(os.path.isfile(export_path))

    def test_ifIsCSVFile_noColumnName_raiseValueError(self):
        create_csv_file(self.fake_data_path_csv, predict_container=True)

        with self.assertRaises(ValueError):
            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.fake_data_path_csv,
                    self.csv_export_filename,
                    "--device",
                    self.cpu_device,
                ]
            )

    def test_ifIsNotSupportedFile_raiseValueError(self):
        create_csv_file(self.fake_data_path_csv, predict_container=True)

        with self.assertRaises(ValueError):
            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.a_unsupported_data_path,
                    self.csv_export_filename,
                    "--device",
                    self.cpu_device,
                ]
            )

    def test_ifIsNotSupportedExportFile_raiseValueError(self):
        create_csv_file(self.fake_data_path_csv, predict_container=True)

        with self.assertRaises(ValueError):
            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.fake_data_path_csv,
                    self.a_unsupported_data_path,
                    "--device",
                    self.cpu_device,
                ]
            )

    def test_ifPathToFakeRetrainModel_thenUseFakeRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            create_pickle_file(self.fake_data_path_pickle, predict_container=True)

            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.fake_data_path_pickle,
                    self.pickle_p_export_filename,
                    "--device",
                    self.cpu_device,
                    "--path_to_retrained_model",
                    self.path_to_retrain_fasttext,
                ]
            )

        expected_first_message = (
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser PreTrainedFastTextAddressParser"
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

    def test_ifPathToFastTextRetrainModel_thenUseFastTextRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            path_to_retrained_model = self.path_to_retrain_fasttext
            create_pickle_file(self.fake_data_path_pickle, predict_container=True)

            parse.main(
                [
                    self.a_fasttext_model_type,
                    self.fake_data_path_pickle,
                    self.pickle_p_export_filename,
                    "--device",
                    self.cpu_device,
                    "--path_to_retrained_model",
                    path_to_retrained_model,
                ]
            )

        expected_first_message = (
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"PreTrainedFastTextAddressParser"
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

    def test_ifPathToBPEmbRetrainModel_thenUseBPEmbRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            create_pickle_file(self.fake_data_path_pickle, predict_container=True)

            parse.main(
                [
                    self.a_bpemb_model_type,
                    self.fake_data_path_pickle,
                    self.pickle_p_export_filename,
                    "--device",
                    self.cpu_device,
                ]
            )

        expected_first_message = (
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"PreTrainedBPEmbAddressParser"
        )

        # Not the same position as with fasttext due to BPEmb messages
        actual_first_message = self._caplog.records[2].message
        self.assertEqual(expected_first_message, actual_first_message)

    def test_ifCachePath_thenUseNewCachePath(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        with patch("deepparse.cli.parse.AddressParser") as address_parser_mock:
            parse.main(
                [
                    self.a_bpemb_model_type,
                    self.fake_data_path_pickle,
                    self.pickle_p_export_filename,
                    "--device",
                    self.cpu_device,
                    "--cache_dir",
                    self.a_cache_dir,
                ]
            )
            address_parser_mock.assert_called()
            address_parser_mock.assert_called_with(
                device=self.cpu_device, cache_dir=self.a_cache_dir, model_type=self.a_bpemb_model_type
            )


if __name__ == "__main__":
    unittest.main()
