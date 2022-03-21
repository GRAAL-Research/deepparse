# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

import argparse
import logging
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import pytest
import torch

from deepparse.cli import parse, generate_export_path, bool_parse
from tests.parser.base import PretrainedWeightsBase
from tests.tools import create_pickle_file, create_csv_file


class ParseTests(TestCase, PretrainedWeightsBase):
    @classmethod
    def setUpClass(cls):
        cls.download_pre_trained_weights(cls)

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()

        self.fake_data_path_pickle = os.path.join(self.temp_dir_obj.name, "fake_data.p")

        self.fake_data_path_csv = os.path.join(self.temp_dir_obj.name, "fake_data.csv")
        self.a_unsupported_data_path = os.path.join(self.temp_dir_obj.name, "fake_data.txt")
        self.fake_data_path_json = os.path.join(self.temp_dir_obj.name, "fake_data.json")

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

        self.create_parser()

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "parsing_model",
            choices=[
                self.a_fasttext_model_type,
                self.a_fasttext_att_model_type,
                self.a_fasttext_light_model_type,
                self.a_bpemb_model_type,
                self.a_bpemb_att_model_type,
            ],
        )

        self.parser.add_argument("dataset_path", type=str)

        self.parser.add_argument("export_filename", type=str)

        self.parser.add_argument("--device", type=str, default="0")

        self.parser.add_argument("--path_to_retrained_model", type=str, default=None)

        self.parser.add_argument("--csv_column_name", type=str, default=None)

        self.parser.add_argument("--csv_column_separator", type=str, default="\t")

        self.parser.add_argument("--log", type=bool_parse, default="True")

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

    @skipIf(not torch.cuda.is_available(), "no gpu available")
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"FastTextAddressParser"
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

        export_path = generate_export_path(self.fake_data_path_pickle, "a_file.p")
        expected_second_message = (
            f"4 addresses have been parsed.\n" f"The parsed addresses are outputted here: {export_path}"
        )
        actual_second_message = self._caplog.records[1].message
        self.assertEqual(expected_second_message, actual_second_message)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

    @skipIf(not torch.cuda.is_available(), "no gpu available")
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
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

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_ifPathToFakeRetrainModel_thenUseFakeRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            # We use the default path to fasttext model as a "retrain model path"
            path_to_retrained_model = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.ckpt")
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
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"FastTextAddressParser"
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"FastTextAddressParser"
        )
        actual_first_message = self._caplog.records[0].message
        self.assertEqual(expected_first_message, actual_first_message)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_ifPathToBPEmbRetrainModel_thenUseBPEmbRetrainModel(self):
        with self._caplog.at_level(logging.INFO):
            path_to_retrained_model = self.path_to_retrain_bpemb
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
            f"Parsing dataset file {self.fake_data_path_pickle} using the parser " f"BPEmbAddressParser"
        )

        # Not the same position as with fasttext due to BPEmb messages
        actual_first_message = self._caplog.records[2].message
        self.assertEqual(expected_first_message, actual_first_message)


if __name__ == "__main__":
    unittest.main()
