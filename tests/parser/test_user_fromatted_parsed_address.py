# pylint: disable=no-member, too-many-public-methods

import io
import sys
import unittest
from unittest import TestCase

from deepparse.parser import FormattedParsedAddress, formatted_parsed_address


class UserFormattedParsedAddressTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_address_str = "3 test road"
        cls.a_parsed_address = [("3", "ATag"), ("test", "AnotherTag"), ("road", "AnotherTag")]
        cls.a_address_repr = "FormattedParsedAddress<ATag='3', AnotherTag='test road'>"
        cls.a_address = {cls.a_address_str: cls.a_parsed_address}
        cls.a_existing_tag = "3"

        cls.a_parsed_address_in_dict_format = {'ALastTag': None, 'ATag': '3', 'AnotherTag': 'test road'}

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out

    def setUp(self):
        # We set the FIELDS of the address base on the prediction tags
        formatted_parsed_address.FIELDS = ["ATag", "AnotherTag", "ALastTag"]

        self.parsed_address = FormattedParsedAddress(self.a_address)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectRawAddress(self):
        address = self.parsed_address.raw_address

        self.assertEqual(address, self.a_address_str)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectParsedAddress(self):
        parsed_address = self.parsed_address.address_parsed_components

        self.assertEqual(parsed_address, self.a_parsed_address)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectTagIfExists(self):
        street_number = self.parsed_address.ATag

        self.assertEqual(street_number, self.a_existing_tag)

    def test_whenInstantiatedWithAddress_thenShouldReturnNoneIfTagDoesntExist(self):
        unit = self.parsed_address.ALastTag

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

    def test_whenToDictUserFields_thenReturnTheProperDict(self):
        actual = self.parsed_address.to_dict(fields=["ATag"])
        expected = {'ATag': '3'}
        self.assertEqual(actual, expected)

    def test_whenFormattedAddressUpperCaseFields_thenReturnAddressWithFieldsUpperCase(self):
        actual = self.parsed_address.format_address(upper_case_fields=["AnotherTag"])
        expected = "3 TEST ROAD"

        self.assertEqual(actual, expected)

    def test_whenFormattedAddressUpperCaseFieldsNotAddressFields_thenRaiseError(self):
        with self.assertRaises(KeyError):
            self.parsed_address.format_address(upper_case_fields=["not_a_field"])

    def test_whenEqualParsedAddress_then__eq__ReturnTrue(self):
        self.assertTrue(self.parsed_address == self.parsed_address)

    def test_whenNotEqualParsedAddressNotSameElements_then__eq__ReturnFalse(self):
        a_different_address_str = "3 test road unit west city province postal_code delivery"

        an_address_with_different_components_tags = [("3", "StreetNumber"), ("test", "StreetName"),
                                                     ("road", "StreetName"), ("unit", "Unit"), ("west", "Orientation"),
                                                     ("city", "Municipality"), ("province", "Province"),
                                                     ("postal_code", "PostalCode"), ("delivery", "GeneralDelivery")]
        another_address = {a_different_address_str: an_address_with_different_components_tags}

        # We reset the FIELDS of the address to default values since we change it in setup
        formatted_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery"
        ]
        different_parsed_address = FormattedParsedAddress(another_address)

        self.assertFalse(self.parsed_address == different_parsed_address)


if __name__ == "__main__":
    unittest.main()
