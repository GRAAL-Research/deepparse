# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

import argparse
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import torch

from deepparse.cli import parse, generate_export_path
from tests.tools import create_pickle_file, create_csv_file


class ParseTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()

        self.fake_data_path_pickle = os.path.join(self.temp_dir_obj.name, "fake_data.p")

        self.fake_data_path_csv = os.path.join(self.temp_dir_obj.name, "fake_data.csv")
        self.a_unsupported_data_path = os.path.join(self.temp_dir_obj.name, "fake_data.txt")

        self.pickle_p_export_file_name = "a_file.p"
        self.pickle_pickle_export_file_name = "a_file.pickle"
        self.csv_export_file_name = "a_file.csv"

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

        self.parser.add_argument("export_file_name", type=str)

        self.parser.add_argument("--device", type=str, default="0")

        self.parser.add_argument("--path_to_retrained_model", type=str, default=None)

        self.parser.add_argument("--csv_column_name", type=str, default=None)

        self.parser.add_argument("--csv_column_separator", type=str, default="\t")

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
                self.pickle_p_export_file_name,
                "--device",
                self.cpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_file_name)
        self.assertTrue(os.path.isfile(export_path))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_integration_gpu(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_model_type,
                self.fake_data_path_pickle,
                self.pickle_p_export_file_name,
                "--device",
                self.gpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_file_name)
        self.assertTrue(os.path.isfile(export_path))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_integration_attention_model(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_att_model_type,
                self.fake_data_path_pickle,
                self.pickle_p_export_file_name,
                "--device",
                self.cpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_file_name)
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
                self.csv_export_file_name,
                "--device",
                self.cpu_device,
                "--csv_column_name",
                "Address",
            ]
        )

        export_path = generate_export_path(self.fake_data_path_csv, self.csv_export_file_name)
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
                self.csv_export_file_name,
                "--device",
                self.cpu_device,
                "--csv_column_name",
                "Address",
                "--csv_column_separator",
                sep,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.csv_export_file_name)
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
                    self.csv_export_file_name,
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
                    self.csv_export_file_name,
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
    def test_ifPathToRetrainModel_thenUseRetrainModel(self):
        create_pickle_file(self.fake_data_path_pickle, predict_container=True)

        parse.main(
            [
                self.a_fasttext_model_type,
                self.fake_data_path_pickle,
                self.pickle_p_export_file_name,
                "--device",
                self.cpu_device,
            ]
        )

        export_path = generate_export_path(self.fake_data_path_pickle, self.pickle_p_export_file_name)
        self.assertTrue(os.path.isfile(export_path))


if __name__ == "__main__":
    unittest.main()
