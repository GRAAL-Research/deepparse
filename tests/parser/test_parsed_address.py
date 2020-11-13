import unittest
from unittest import TestCase

from deepparse.parser import ParsedAddress


class ParsedAddressTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_address_str = "3 test road"
        cls.a_parsed_address = [("3", "StreetNumber"), ("test", "StreetName"), ("road", "StreetName")]
        cls.a_address = {cls.a_address_str: cls.a_parsed_address}
        cls.a_existing_tag = "3"

    def setUp(self):
        self.parsed_address = ParsedAddress(self.a_address)

    def test_whenInstanciatedWithAddress_thenShouldReturnCorrectRawAddress(self):
        address = self.parsed_address.raw_address

        self.assertEqual(address, self.a_address_str)

    def test_whenInstanciatedWithAddress_thenShouldReturnCorrectParsedAddress(self):
        parsed_address = self.parsed_address.address_parsed_components

        self.assertEqual(parsed_address, self.a_parsed_address)

    def test_whenInstanciatedWithAddress_thenShouldReturnCorrectTagIfExists(self):
        street_number = self.parsed_address.street_number

        self.assertEqual(street_number, self.a_existing_tag)

    def test_whenInstanciatedWithAddress_thenShouldReturnNoneIfTagDoesntExist(self):
        unit = self.parsed_address.unit

        self.assertIsNone(unit)


if __name__ == "__main__":
    unittest.main()
