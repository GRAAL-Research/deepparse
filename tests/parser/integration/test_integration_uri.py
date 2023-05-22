# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from unittest import skipIf

from tests.parser.integration.base_predict import (
    AddressParserPredictBase,
)


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class AddressParserPredictURITest(AddressParserPredictBase):
    @classmethod
    def setUpClass(cls):
        super(AddressParserPredictURITest, cls).setUpClass()
        cls.device = "cpu"

    def test_givenAAddress_whenParseFastTextURI_thenParseAddress(self):
        config = {
            "model_type": "fasttext",
            "device": self.device,
            "verbose": False,
            "path_to_retrained_model": "s3://deepparse/fasttext.ckpt",
        }
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse)
        self.assert_properly_parse(parse_address)
