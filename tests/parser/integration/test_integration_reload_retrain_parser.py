# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, too-many-public-methods

import os
from unittest import skipIf

from deepparse.parser import AddressParser
from tests.parser.base import PretrainedWeightsBase
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class AddressParserIntegrationReloadRetrainAPITest(AddressParserRetrainTestCase, PretrainedWeightsBase):
    @classmethod
    def setUpClass(cls):
        super(AddressParserIntegrationReloadRetrainAPITest, cls).setUpClass()
        cls.download_pre_trained_weights(cls)

    def test_integration_parsing_with_retrain_fasttext(self):
        model_type = "fasttext"
        path_to_retrained_model = self.path_to_retrain_fasttext

        address_parser = AddressParser(model_type=model_type, path_to_retrained_model=path_to_retrained_model)
        self.assertEqual(model_type, address_parser.model_type)

    def test_integration_parsing_with_retrain_bpemb(self):
        model_type = "bpemb"
        path_to_retrained_model = self.path_to_retrain_bpemb

        address_parser = AddressParser(model_type=model_type, path_to_retrained_model=path_to_retrained_model)
        self.assertEqual(model_type, address_parser.model_type)

    def test_integration_parsing_with_retrain_named_model(self):
        model_type = "fasttext"  # Base architecture of the named model is a FastText
        path_to_retrained_model = self.path_to_named_model

        address_parser = AddressParser(model_type=model_type, path_to_retrained_model=path_to_retrained_model)
        self.assertEqual(model_type, address_parser.model_type)

        expected_name = "MyModelNewName"
        actual_name = str(address_parser)
        self.assertEqual(expected_name, actual_name)
