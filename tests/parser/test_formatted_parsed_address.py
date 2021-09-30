# pylint: disable=no-member, too-many-public-methods

import io
import sys
import unittest
from unittest import TestCase

from deepparse.parser import FormattedParsedAddress, formatted_parsed_address


class FormattedParsedAddressTest(TestCase):

    @classmethod
    def setUpClass(cls):
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
        # we reset the FIELDS of the address to default values since we change it in some tests
        formatted_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery"
        ]

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

    def test_whenFormattedAddressDefaultSettings_thenReturnExpectedOrderAndDontReturnNoneComponents(self):
        actual = self.parsed_address.format_address()
        expected = self.a_address_str

        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.format_address()
        expected = "3 test road unit west city province postal_code delivery"

        self.assertEqual(actual, expected)

    def test_whenFormattedAddressFieldsChanged_thenReturnNewOrderFields(self):
        a_different_order = [
            "GeneralDelivery", "Unit", "StreetName", "StreetNumber", "Orientation", "Municipality", "Province",
            "PostalCode"
        ]
        actual = self.parsed_address.format_address(fields=a_different_order)
        expected = "test road 3"

        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.format_address(fields=a_different_order)
        expected = "delivery unit test road 3 west city province postal_code"

        self.assertEqual(actual, expected)

    def test_whenFormattedAddressFieldsSeparator_thenReturnAddressWithFieldsSeparator(self):
        actual = self.parsed_address.format_address(field_separator=", ")
        expected = "3, test road"

        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.format_address(field_separator=", ")
        expected = "3, test road, unit, west, city, province, postal_code, delivery"

        self.assertEqual(actual, expected)

    def test_whenFormattedAddressCapitalizeFields_thenReturnAddressWithFieldsCapitalize(self):
        actual = self.parsed_address.format_address(capitalize_fields=["StreetName"])
        expected = "3 Test road"

        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.format_address(capitalize_fields=["PostalCode", "Province"])
        expected = "3 test road unit west city Province Postal_code delivery"

        self.assertEqual(actual, expected)

    def test_whenFormattedAddressCapitalizeFieldsNotAddressFields_thenRaiseError(self):
        with self.assertRaises(KeyError):
            self.parsed_address.format_address(capitalize_fields=["not_a_field"])

        with self.assertRaises(KeyError):
            self.complete_parsed_address.format_address(capitalize_fields=["not_a_field"])

    def test_whenFormattedAddressUpperCaseFields_thenReturnAddressWithFieldsUpperCase(self):
        actual = self.parsed_address.format_address(upper_case_fields=["StreetName"])
        expected = "3 TEST ROAD"

        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.format_address(upper_case_fields=["PostalCode", "Province"])
        expected = "3 test road unit west city PROVINCE POSTAL_CODE delivery"

        self.assertEqual(actual, expected)

    def test_whenFormattedAddressUpperCaseFieldsNotAddressFields_thenRaiseError(self):
        with self.assertRaises(KeyError):
            self.parsed_address.format_address(upper_case_fields=["not_a_field"])

        with self.assertRaises(KeyError):
            self.complete_parsed_address.format_address(upper_case_fields=["not_a_field"])

    def test_whenFormattedAddressAllArgsChanged_thenReturnAddressProperlyFormatted(self):
        a_different_order = [
            "GeneralDelivery", "Unit", "StreetName", "StreetNumber", "Orientation", "Municipality", "Province",
            "PostalCode"
        ]

        actual = self.complete_parsed_address.format_address(fields=a_different_order,
                                                             field_separator=", ",
                                                             capitalize_fields=["StreetName"],
                                                             upper_case_fields=["PostalCode", "Province"])
        expected = "delivery, unit, Test road, 3, west, city, PROVINCE, POSTAL_CODE"

        self.assertEqual(actual, expected)

    def test_whenFormattedParsedAddressInferredOrder_thenProperlyInferred(self):
        actual = self.parsed_address.inferred_order
        expected = ['StreetNumber', 'StreetName']

        self.assertEqual(actual, expected)

        actual = self.complete_parsed_address.inferred_order
        expected = [
            'StreetNumber', 'StreetName', 'Unit', 'Orientation', 'Municipality', 'Province', 'PostalCode',
            'GeneralDelivery'
        ]

        self.assertEqual(actual, expected)

    def test_whenEqualParsedAddress_then__eq__ReturnTrue(self):
        self.assertTrue(self.parsed_address == self.parsed_address)
        self.assertTrue(self.complete_parsed_address == self.complete_parsed_address)

    def test_whenNotEqualParsedAddress_then__eq__ReturnFalse(self):
        self.assertFalse(self.parsed_address == self.complete_parsed_address)
        self.assertFalse(self.complete_parsed_address == self.parsed_address)


if __name__ == "__main__":
    unittest.main()
