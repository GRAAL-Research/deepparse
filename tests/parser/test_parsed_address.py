import unittest

from deepparse.parser import ParsedAddress
from tests.base_capture_output import CaptureOutputTestCase


class ParsedAddressTest(CaptureOutputTestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_address_str = "3 test road"
        cls.a_complete_address_str = "3 test road unit west city province postal_code delivery"
        cls.a_parsed_address = [("3", "StreetNumber"), ("test", "StreetName"), ("road", "StreetName")]
        cls.a_complete_parsed_address = [("3", "StreetNumber"), ("test", "StreetName"), ("road", "StreetName"),
                                         ("unit", "Unit"), ("west", "Orientation"), ("city", "Municipality"),
                                         ("province", "Province"), ("postal_code", "PostalCode"),
                                         ("delivery", "GeneralDelivery")]

        cls.a_address_repr = "ParsedAddress<street_number='3', street_name='test road'>"
        cls.a_address = {cls.a_address_str: cls.a_parsed_address}
        cls.a_complete_address = {cls.a_complete_address_str: cls.a_complete_parsed_address}
        cls.a_existing_tag = "3"

    def setUp(self):
        self.parsed_address = ParsedAddress(self.a_address)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectRawAddress(self):
        address = self.parsed_address.raw_address

        self.assertEqual(address, self.a_address_str)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectParsedAddress(self):
        parsed_address = self.parsed_address.address_parsed_components

        self.assertEqual(parsed_address, self.a_parsed_address)

    def test_whenInstantiatedWithCompleteAddress_thenShouldReturnCorrectCompleteParsedAddress(self):
        self.parsed_address = ParsedAddress(self.a_complete_address)
        parsed_address = self.parsed_address.address_parsed_components

        self.assertEqual(parsed_address, self.a_complete_parsed_address)

    def test_whenInstantiatedWithAddress_thenShouldReturnCorrectTagIfExists(self):
        street_number = self.parsed_address.street_number

        self.assertEqual(street_number, self.a_existing_tag)

    def test_whenInstantiatedWithAddress_thenShouldReturnNoneIfTagDoesntExist(self):
        unit = self.parsed_address.unit

        self.assertIsNone(unit)

    def test_whenStrAnParseAddress_thenStringIsTheRawAddress(self):
        self._capture_output()

        print(self.parsed_address)

        self.assertEqual(self.a_address_str, self.test_out.getvalue().strip())

    def test_whenReprAnParseAddress_thenStringIsTheAddressFormatted(self):
        self._capture_output()

        print(self.parsed_address.__repr__())

        self.assertEqual(self.a_address_repr, self.test_out.getvalue().strip())


if __name__ == "__main__":
    unittest.main()
