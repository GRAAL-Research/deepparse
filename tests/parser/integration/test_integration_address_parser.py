from typing import List
from unittest import TestCase

from deepparse.parser import AddressParser, FormattedParsedAddress


class AddressParserPredictTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fasttext_address_parser = AddressParser(model_type="fasttext", device=0)
        cls.bpemb_address_parser = AddressParser(model_type="bpemb", device=0)

    def setUp(self):
        self.an_address_to_parse = "350 rue des lilas o"

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
