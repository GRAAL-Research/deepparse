# pylint: disable=no-member

import io
import sys
import unittest
from unittest import TestCase

from deepparse.parser import FormattedParsedAddress


class FormattedParsedAddressTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.address_components = [
            "StreetNumber", "StreetName", "Unit", "Municipality", "Province", "PostalCode", "Orientation",
            "GeneralDelivery"
        ]
        cls.a_address_str = "3 test road"
        cls.a_complete_address_str = "3 test road unit west city province postal_code delivery"
        cls.a_parsed_address = [("3", "StreetNumber"), ("test", "StreetName"), ("road", "StreetName")]
        cls.a_complete_parsed_address = [("3", "StreetNumber"), ("test", "StreetName"), ("road", "StreetName"),
                                         ("unit", "Unit"), ("west", "Orientation"), ("city", "Municipality"),
                                         ("province", "Province"), ("postal_code", "PostalCode"),
                                         ("delivery", "GeneralDelivery")]

        cls.a_address_repr = "FormattedParsedAddress<StreetNumber='3', StreetName='test road'>"
        cls.a_address = {cls.a_address_str: cls.a_parsed_address}
        cls.a_complete_address = {cls.a_complete_address_str: cls.a_complete_parsed_address}
        cls.a_existing_tag = "3"

        cls.a_parsed_address_in_dict_format = {
            'StreetNumber': '3',
            'Unit': None,
            'StreetName': 'test road',
            'Orientation': None,
            'Municipality': None,
            'Province': None,
            'PostalCode': None,
            'GeneralDelivery': None
        }

        cls.a_complete_parsed_address_in_dict_format = {
            'StreetNumber': '3',
            'Unit': 'unit',
            'StreetName': 'test road',
            'Orientation': 'west',
            'Municipality': 'city',
            'Province': 'province',
            'PostalCode': 'postal_code',
            'GeneralDelivery': 'delivery'
        }

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out

    def setUp(self):
        self.parsed_address = FormattedParsedAddress(self.a_address)
        self.complete_parsed_address = FormattedParsedAddress(self.a_complete_address)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectRawAddress(self):
        address = self.parsed_address.raw_address

        self.assertEqual(address, self.a_address_str)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectParsedAddress(self):
        parsed_address = self.parsed_address.address_parsed_components

        self.assertEqual(parsed_address, self.a_parsed_address)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectTagIfExists(self):
        street_number = self.parsed_address.StreetNumber

        self.assertEqual(street_number, self.a_existing_tag)

    def test_whenInstantiatedWithAddress_thenShouldReturnNoneIfTagDoesntExist(self):
        unit = self.parsed_address.Unit

        self.assertIsNone(unit)

    def test_whenStrAnParseAddress_thenStringIsTheRawAddress(self):
        self._capture_output()

        print(self.parsed_address)

        self.assertEqual(self.a_address_str, self.test_out.getvalue().strip())

    def test_whenReprAnParseAddress_thenStringIsTheAddressFormatted(self):
        self._capture_output()

        print(self.parsed_address.__repr__())

        self.assertEqual(self.a_address_repr, self.test_out.getvalue().strip())

    def test_whenToDictDefaultFields_thenReturnTheProperDict(self):
        actual = self.parsed_address.to_dict()
        expected = self.a_parsed_address_in_dict_format
        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.to_dict()
        expected = self.a_complete_parsed_address_in_dict_format
        self.assertEqual(actual, expected)

    def test_whenToDictUserFields_thenReturnTheProperDict(self):
        actual = self.parsed_address.to_dict(fields=["StreetNumber"])
        expected = {'StreetNumber': '3'}
        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.to_dict(fields=["StreetNumber"])
        expected = {'StreetNumber': '3'}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
