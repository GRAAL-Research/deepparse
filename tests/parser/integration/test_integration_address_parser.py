# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import skipIf
from unittest.mock import patch

import torch

from tests.base_file_exist import FileCreationTestCase
from tests.parser.integration.base_predict import (
    AddressParserBase,
)


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class AddressParserTest(AddressParserBase, FileCreationTestCase):
    @classmethod
    def setUpClass(cls):
        super(AddressParserTest, cls).setUpClass()

        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_saving_dir_path = cls.temp_dir_obj.name

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def setting_up_default_parser_model(self) -> None:
        a_config = {"model_type": "fasttext", "device": "cpu", "verbose": False}
        self.setup_model_with_config(a_config)

    def test_givenAModelToExportDictStr_thenExportIt(self):
        self.setting_up_default_parser_model()

        a_file_path = os.path.join(self.a_saving_dir_path, "exported_model.p")

        self.a_model.save_model_weights(file_path=a_file_path)

        self.assertFileExist(a_file_path)

    def test_givenAModelToExportDictPathALike_thenExportIt(self):
        self.setting_up_default_parser_model()

        a_file_path = Path(os.path.join(self.a_saving_dir_path, "exported_model.p"))

        self.a_model.save_model_weights(file_path=a_file_path)

        self.assertFileExist(a_file_path)

    def test_givenAnExportedModelUsingTheMethod_whenReloadIt_thenReload(self):
        self.setting_up_default_parser_model()

        a_file_path = Path(os.path.join(self.a_saving_dir_path, "exported_model.p"))

        self.a_model.save_model_weights(file_path=a_file_path)

        weights = torch.load(a_file_path)

        self.assertIsInstance(weights, OrderedDict)

        model_layer_keys = [
            'encoder.lstm.weight_ih_l0',
            'encoder.lstm.weight_hh_l0',
            'encoder.lstm.bias_ih_l0',
            'encoder.lstm.bias_hh_l0',
            'decoder.lstm.weight_ih_l0',
            'decoder.lstm.weight_hh_l0',
            'decoder.lstm.bias_ih_l0',
            'decoder.lstm.bias_hh_l0',
            'decoder.linear.weight',
            'decoder.linear.bias',
        ]
        self.assertEqual(model_layer_keys, list(weights.keys()))

    def test_givenAOfflineAddressParser_whenInitWithLocalFiles_thenDontCallDownloadWeights(self):
        a_model_type = "fasttext"
        a_device = "cpu"

        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.setup_model_with_config({"model_type": a_model_type, "device": a_device, "offline": True})
            download_weights_mock.assert_not_called()
