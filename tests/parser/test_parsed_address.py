import unittest
from unittest import TestCase

from deepparse.parser import ParsedAddress


class ParsedAddressTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A_ADDRESS_STR = '3 test road'
        cls.A_PARSED_ADDRESS = {'3': 'StreetNumber', 'test': 'StreetName', 'road': 'StreetName'}
        cls.A_ADDRESS = {cls.A_ADDRESS_STR: cls.A_PARSED_ADDRESS}
        cls.A_EXISTING_TAG = '3'

    def setUp(self):
        self.parsed_address = ParsedAddress(self.A_ADDRESS)

    def test_whenInstanciatedWithAddress_thenShouldReturnCorrectRawAddress(self):
        address = self.parsed_address.raw_address

        self.assertEqual(address, self.A_ADDRESS_STR)

    def test_whenInstanciatedWithAddress_thenShouldReturnCorrectParsedAddress(self):
        parsed_address = self.parsed_address.address_parsed_dict

        self.assertEqual(parsed_address, self.A_PARSED_ADDRESS)

    def test_whenInstanciatedWithAddress_thenShouldReturnCorrectTagIfExists(self):
        street_number = self.parsed_address.street_number

        self.assertEqual(street_number, self.A_EXISTING_TAG)

    def test_whenInstanciatedWithAddress_thenShouldReturnNoneIfTagDoesntExist(self):
        unit = self.parsed_address.unit

        self.assertIsNone(unit)


if __name__ == '__main__':
    unittest.main()