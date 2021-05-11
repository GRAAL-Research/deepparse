# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from typing import List
from unittest import TestCase, skipIf

import os

from deepparse.parser import AddressParser, ParsedAddress


@skipIf(not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.version"))
        or not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.version")),
        "download of model too long for test in runner")
class AddressParserPredictCPUTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fasttext_address_parser = AddressParser(model_type="fasttext", device='cpu')
        cls.bpemb_address_parser = AddressParser(model_type="bpemb", device='cpu')

    def setUp(self):
        self.an_address_to_parse = "350 rue des lilas o"

    def test_givenAAddress_whenParse_thenParseAddress(self):
        parse_address = self.fasttext_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, ParsedAddress)

        parse_address = self.bpemb_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, ParsedAddress)

    def test_givenAListOfAddress_whenParse_thenParseAllAddress(self):
        parse_address = self.fasttext_address_parser([self.an_address_to_parse, self.an_address_to_parse])
        self.assertIsInstance(parse_address, List)

        parse_address = self.bpemb_address_parser([self.an_address_to_parse, self.an_address_to_parse])
        self.assertIsInstance(parse_address, List)


# test if num_workers > 0 is correct for the data loader
@skipIf(not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.version"))
        or not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.version")),
        "download of model too long for test in runner")
class AddressParserPredictCPUMultiProcessTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fasttext_address_parser = AddressParser(model_type="fasttext", device='cpu')
        cls.bpemb_address_parser = AddressParser(model_type="bpemb", device='cpu')

    def setUp(self):
        self.an_address_to_parse = "350 rue des lilas o"

    def test_givenAAddress_whenParseNumWorkers1_thenParseAddress(self):
        parse_address = self.fasttext_address_parser(self.an_address_to_parse, num_workers=1)
        self.assertIsInstance(parse_address, ParsedAddress)

        parse_address = self.bpemb_address_parser(self.an_address_to_parse, num_workers=1)
        self.assertIsInstance(parse_address, ParsedAddress)

    def test_givenAListOfAddress_whenParseNumWorkers1_thenParseAllAddress(self):
        parse_address = self.fasttext_address_parser([self.an_address_to_parse, self.an_address_to_parse],
                                                     num_workers=1)
        self.assertIsInstance(parse_address, List)

        parse_address = self.bpemb_address_parser([self.an_address_to_parse, self.an_address_to_parse], num_workers=1)
        self.assertIsInstance(parse_address, List)

    def test_givenAAddress_whenParseNumWorkers2_thenParseAddress(self):
        parse_address = self.fasttext_address_parser(self.an_address_to_parse, num_workers=2)
        self.assertIsInstance(parse_address, ParsedAddress)

        parse_address = self.bpemb_address_parser(self.an_address_to_parse, num_workers=2)
        self.assertIsInstance(parse_address, ParsedAddress)

    def test_givenAListOfAddress_whenParseNumWorkers2_thenParseAllAddress(self):
        parse_address = self.fasttext_address_parser([self.an_address_to_parse, self.an_address_to_parse],
                                                     num_workers=2)
        self.assertIsInstance(parse_address, List)

        parse_address = self.bpemb_address_parser([self.an_address_to_parse, self.an_address_to_parse], num_workers=2)
        self.assertIsInstance(parse_address, List)
