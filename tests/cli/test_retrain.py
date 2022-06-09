# pylint: disable=too-many-arguments, too-many-locals

import os
import unittest
from tempfile import TemporaryDirectory
from typing import List
from unittest import skipIf

import torch

from deepparse.cli import retrain
from deepparse.cli.retrain import get_args, parse_retrained_arguments
from tests.parser.integration.base_retrain import RetrainTestCase


class RetrainTests(RetrainTestCase):
    @classmethod
    def setUpClass(cls):
        super(RetrainTests, cls).setUpClass()

        cls.a_fasttext_model_type = "fasttext"
        cls.a_fasttext_att_model_type = "fasttext-attention"
        cls.a_fasttext_light_model_type = "fasttext-light"
        cls.a_bpemb_model_type = "bpemb"
        cls.a_bpemb_att_model_type = "bpemd-attention"

        cls.cpu_device = "cpu"
        cls.gpu_device = "0"

        cls.a_named_model = "a_retrained_model"

    def setUp(self) -> None:
        self.temp_checkpoints_obj = TemporaryDirectory()

        self.export_named_retrain_path = os.path.join(self.temp_checkpoints_obj.name, self.a_named_model + ".ckpt")

        # We use the default checkpoints logging path but redirect it into the
        # temp directory
        self.logging_path = os.path.join(self.temp_checkpoints_obj.name, "checkpoints")

        # We set a set of defaults parser argument to adjust it for the tests
        # We change the default logging path to the one created as a temp
        # directory for cleanup, and we reduce the number of epoch tho reduce
        # test duration.
        self.parser_test_default_settings = ["--logging_path", self.logging_path, "--epochs", "1"]

    def tearDown(self) -> None:
        self.temp_checkpoints_obj.cleanup()

    def set_up_params(
        self,
        train_ratio="0.8",
        batch_size="32",
        epochs="5",
        num_workers="1",
        learning_rate="0.01",
        seed="42",
        logging_path="./checkpoints",
        disable_tensorboard="False",
        layers_to_freeze='seq2seq',
        name_of_the_retrain_parser="",
    ) -> List:
        parser_params = [
            self.a_fasttext_model_type,
            self.a_train_pickle_dataset_path,
            "--train_ratio",
            train_ratio,
            "--batch_size",
            batch_size,
            "--epochs",
            epochs,
            "--num_workers",
            num_workers,
            "--learning_rate",
            learning_rate,
            "--seed",
            seed,
            "--logging_path",
            logging_path,
            "--disable_tensorboard",
            disable_tensorboard,
            "--layers_to_freeze",
            layers_to_freeze,
            "--name_of_the_retrain_parser",
            name_of_the_retrain_parser,
        ]
        return parser_params

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_integration_cpu(self):
        parser_params = [
            self.a_fasttext_model_type,
            self.a_train_pickle_dataset_path,
            "--device",
            self.cpu_device,
        ]
        parser_params.extend(self.parser_test_default_settings)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_integration_gpu(self):
        parser_params = [
            self.a_fasttext_model_type,
            self.a_train_pickle_dataset_path,
            "--device",
            self.gpu_device,
        ]

        parser_params.extend(self.parser_test_default_settings)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_integration_attention_model(self):
        parser_params = [
            self.a_fasttext_att_model_type,
            self.a_train_pickle_dataset_path,
            "--device",
            self.cpu_device,
        ]

        parser_params.extend(self.parser_test_default_settings)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_attention_address_parser.ckpt"
                )
            )
        )

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
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

        parser_params.extend(self.parser_test_default_settings)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_ifIsCSVFile_noColumnName_raiseValueError(self):
        with self.assertRaises(ValueError):
            parser_params = [
                self.a_fasttext_model_type,
                self.a_train_csv_dataset_path,
                "--device",
                self.cpu_device,
            ]

            parser_params.extend(self.parser_test_default_settings)
            retrain.main(parser_params)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_ifIsNotSupportedFile_raiseValueError(self):
        with self.assertRaises(ValueError):
            parser_params = [
                self.a_fasttext_model_type,
                "an_unsupported_extension.json",
                "--device",
                self.cpu_device,
            ]

            parser_params.extend(self.parser_test_default_settings)
            retrain.main(parser_params)

    def test_givenSetOfParams_whenParseTrainingParams_thenReturnProperParams(self):
        override_keys = ["train_ratio", "batch_size"]
        train_ratio = "0.8"
        batch_size = "64"
        override_values = [float(train_ratio), int(batch_size)]
        parser_params = self.set_up_params(train_ratio=train_ratio, batch_size=batch_size)

        parsed_args = get_args(parser_params)

        parse_retrained_args = parse_retrained_arguments(parsed_args)

        for key, value in parse_retrained_args.items():
            if key in override_keys:
                self.assertTrue(value in override_values)

        override_keys = ["num_workers", "epochs"]
        num_workers = "2"
        epochs = "64"
        override_values = [int(num_workers), int(epochs)]
        parser_params = self.set_up_params(num_workers=num_workers, epochs=epochs)

        parsed_args = get_args(parser_params)

        parse_retrained_args = parse_retrained_arguments(parsed_args)

        for key, value in parse_retrained_args.items():
            if key in override_keys:
                self.assertTrue(value in override_values)

        override_keys = ["disable_tensorboard", "layers_to_freeze", "name_of_the_retrain_parser", "logging_path"]
        disable_tensorboard = "True"
        layers_to_freeze = "encoder"
        name_of_the_retrain_parser = "AName"
        logging_path = "APath/"
        override_values = [layers_to_freeze, bool(disable_tensorboard), name_of_the_retrain_parser, logging_path]
        parser_params = self.set_up_params(
            layers_to_freeze=layers_to_freeze,
            disable_tensorboard=disable_tensorboard,
            name_of_the_retrain_parser=name_of_the_retrain_parser,
            logging_path=logging_path,
        )

        parsed_args = get_args(parser_params)

        parse_retrained_args = parse_retrained_arguments(parsed_args)

        for key, value in parse_retrained_args.items():
            if key in override_keys:
                self.assertTrue(value in override_values)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_integration_multiple_retrain_overrides(self):
        train_ratio = "0.8"
        batch_size = "64"
        num_workers = "2"
        epochs = "2"
        disable_tensorboard = "True"
        layers_to_freeze = "encoder"
        name_of_the_retrain_parser = "AName"
        logging_path = self.logging_path

        parser_params = self.set_up_params(
            train_ratio=train_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            layers_to_freeze=layers_to_freeze,
            disable_tensorboard=disable_tensorboard,
            name_of_the_retrain_parser=name_of_the_retrain_parser,
            logging_path=logging_path,
        )
        retrain.main(parser_params)

        self.assertTrue(os.path.isfile(os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "AName.ckpt")))


if __name__ == "__main__":
    unittest.main()
