# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import skipIf

import torch

from tests.base_file_exist import FileCreationTestCase
from tests.parser.integration.base_predict import (
    AddressParserBase,
)


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class AddressParserTest(AddressParserBase, FileCreationTestCase):
    @classmethod
    def setUpClass(cls):
        super(AddressParserTest, cls).setUpClass()

        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_saving_dir_path = cls.temp_dir_obj.name

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def setUp(self) -> None:
        a_config = {"model_type": "fasttext", "device": "cpu", "verbose": False}
        self.setup_model_with_config(a_config)

    def test_givenAModelToExportDictStr_thenExportIt(self):
        a_file_path = os.path.join(self.a_saving_dir_path, "exported_model.p")

        self.a_model.save_model_weights(file_path=a_file_path)

        self.assertFileExist(a_file_path)

    def test_givenAModelToExportDictPathALike_thenExportIt(self):
        a_file_path = Path(os.path.join(self.a_saving_dir_path, "exported_model.p"))

        self.a_model.save_model_weights(file_path=a_file_path)

        self.assertFileExist(a_file_path)

    def test_givenAnExportedModelUsingTheMethod_whenReloadIt_thenReload(self):
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
