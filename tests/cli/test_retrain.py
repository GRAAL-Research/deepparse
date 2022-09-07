# pylint: disable=too-many-arguments, too-many-locals

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from tempfile import TemporaryDirectory
from typing import List
from unittest import skipIf
from unittest.mock import patch

from deepparse.cli import retrain
from deepparse.cli.retrain import get_args, parse_retrained_arguments
from tests.parser.integration.base_retrain import RetrainTestCase


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
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

        cls.a_cache_dir = "a_cache_dir"

    def setUp(self) -> None:
        self.temp_checkpoints_obj = TemporaryDirectory()

        self.export_named_retrain_path = os.path.join(self.temp_checkpoints_obj.name, self.a_named_model + ".ckpt")

        # We use the default checkpoints logging path but redirect it into the
        # temp directory
        self.logging_path = os.path.join(self.temp_checkpoints_obj.name, "checkpoints")

    def tearDown(self) -> None:
        self.temp_checkpoints_obj.cleanup()

    def set_up_params(
        self,
        model_type=None,  # None to handle the default tests case.
        train_dataset_path=None,  # None to handle the default tests case.
        val_dataset_path=None,
        train_ratio="0.8",
        batch_size="32",
        epochs="1",  # As opposed to default CLI function, we set the epoch value number to 1,
        num_workers="1",
        learning_rate="0.01",
        seed="42",
        logging_path=None,  # None to handle the default case logging path for the tests.
        disable_tensorboard="False",
        layers_to_freeze='seq2seq',
        name_of_the_retrain_parser=None,
        cache_dir=None,  # None to handle the default library default value (CACHE_PATH)
        device="cpu",  # By default, we set it to cpu instead of gpu device 0 as the CLI function.
        csv_column_names: List = None,
        csv_column_separator="\t",
    ) -> List:
        if model_type is None:
            # The default case for the test is a FastText model
            model_type = self.a_fasttext_model_type

        if train_dataset_path is None:
            train_dataset_path = self.a_train_pickle_dataset_path

        if logging_path is None:
            logging_path = self.logging_path

        parser_params = [
            model_type,
            train_dataset_path,
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
            "--device",
            device,
            "--csv_column_separator",
            csv_column_separator,
        ]

        if val_dataset_path is not None:
            # To handle the None case (that is using the default None of the argparser).
            parser_params.extend(["--val_dataset_path", val_dataset_path])

        if cache_dir is not None:
            # To handle the None case (that is using the default None of the argparser).
            parser_params.extend(["--cache_dir", cache_dir])

        if name_of_the_retrain_parser is not None:
            # To handle the None case (that is using the default None of the argparser).
            parser_params.extend(["--name_of_the_retrain_parser", name_of_the_retrain_parser])

        if csv_column_names is not None:
            # To handle the None case (that is using the default None of the argparser).
            parser_params.extend(["--csv_column_names"])
            parser_params.extend(csv_column_names)  # Since csv_column_names is a list

        return parser_params

    def test_integration_cpu(self):
        parser_params = self.set_up_params(device=self.cpu_device)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )

    def test_integration_gpu(self):
        parser_params = self.set_up_params(device=self.gpu_device)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )

    def test_integration_attention_model(self):
        parser_params = self.set_up_params(model_type=self.a_fasttext_att_model_type)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_attention_address_parser.ckpt"
                )
            )
        )

    def test_integration_csv(self):
        parser_params = self.set_up_params(
            train_dataset_path=self.a_train_csv_dataset_path,
            csv_column_names=["Address", "Tags"],
            csv_column_separator=",",
        )

        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )

    def test_ifIsCSVFile_noColumnName_raiseValueError(self):
        with self.assertRaises(ValueError):
            # We set up the params with the default value of csv_column_names of the test case method set_up_params,
            # which is None, thus no column names.
            parser_params = self.set_up_params(train_dataset_path=self.a_train_csv_dataset_path)

            retrain.main(parser_params)

    def test_ifIsNotSupportedFile_raiseValueError(self):
        with self.assertRaises(ValueError):
            parser_params = self.set_up_params(train_dataset_path="an_unsupported_extension.json")

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

    def test_ifCachePath_thenUseNewCachePath(self):
        with patch("deepparse.cli.retrain.AddressParser") as address_parser_mock:
            parser_params = self.set_up_params(cache_dir=self.a_cache_dir, epochs="1")
            retrain.main(parser_params)

            address_parser_mock.assert_called()
            address_parser_mock.assert_called_with(
                device=self.cpu_device, cache_dir=self.a_cache_dir, model_type=self.a_fasttext_model_type
            )  # Default tests case default model type is the FastText model

    def test_integrationWithValDataset(self):
        parser_params = self.set_up_params(device=self.cpu_device, val_dataset_path=self.a_train_pickle_dataset_path)
        retrain.main(parser_params)

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.temp_checkpoints_obj.name, "checkpoints", "retrained_fasttext_address_parser.ckpt")
            )
        )


if __name__ == "__main__":
    unittest.main()
