# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from typing import List
from unittest import skipIf

import torch

from deepparse.parser import FormattedParsedAddress
from tests.parser.integration.base_predict import AddressParserPredictBase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserPredictGPUTest(AddressParserPredictBase):

    def test_givenAAddress_whenParse_thenParseAddress(self):
        parse_address = self.fasttext_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

        parse_address = self.bpemb_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

    def test_givenAListOfAddress_whenParse_thenParseAllAddress(self):
        parse_address = self.fasttext_address_parser([self.an_address_to_parse, self.an_address_to_parse])
        self.assertIsInstance(parse_address, List)

        parse_address = self.bpemb_address_parser([self.an_address_to_parse, self.an_address_to_parse])
        self.assertIsInstance(parse_address, List)


# test if num_workers > 0 is correct for the data loader
@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserPredictGPUMultiProcessTest(AddressParserPredictBase):

    def test_givenAAddress_whenParseNumWorkers1_thenParseAddress(self):
        parse_address = self.fasttext_address_parser(self.an_address_to_parse, num_workers=1)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

        parse_address = self.bpemb_address_parser(self.an_address_to_parse, num_workers=1)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

    def test_givenAListOfAddress_whenParseNumWorkers1_thenParseAllAddress(self):
        parse_address = self.fasttext_address_parser([self.an_address_to_parse, self.an_address_to_parse],
                                                     num_workers=1)
        self.assertIsInstance(parse_address, List)

        parse_address = self.bpemb_address_parser([self.an_address_to_parse, self.an_address_to_parse], num_workers=1)
        self.assertIsInstance(parse_address, List)

    def test_givenAAddress_whenParseNumWorkers2_thenParseAddress(self):
        parse_address = self.fasttext_address_parser(self.an_address_to_parse, num_workers=2)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

        parse_address = self.bpemb_address_parser(self.an_address_to_parse, num_workers=2)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

    def test_givenAListOfAddress_whenParseNumWorkers2_thenParseAllAddress(self):
        parse_address = self.fasttext_address_parser([self.an_address_to_parse, self.an_address_to_parse],
                                                     num_workers=2)
        self.assertIsInstance(parse_address, List)

        parse_address = self.bpemb_address_parser([self.an_address_to_parse, self.an_address_to_parse], num_workers=2)
        self.assertIsInstance(parse_address, List)
