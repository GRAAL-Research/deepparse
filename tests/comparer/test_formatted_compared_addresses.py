# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from deepparse.comparer import AddressesComparer
from deepparse.parser.address_parser import AddressParser
from tests.base_capture_output import CaptureOutputTestCase


class TestAddressComparer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_addresses_comparer_model = "bpemb"
        cls.a_addresses_comparer__repr__ = f"Compare addresses with {cls.a_addresses_comparer_model.capitalize()}AddressParser"

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
        formated_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery"
        ]

    def setUp(self) -> None:
        self.raw_address_original = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_identical = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_equivalent = "350  rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_diff_streetNumber = "450 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_diff_streetName = "350 Boulevard des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_diff_Unit = "350 rue des Lilas Ouest app 105 Québec Québec G1L 1B6"
        self.raw_address_diff_Municipality = "350 rue des Lilas Ouest Ste-Foy Québec G1L 1B6"
        self.raw_address_diff_Province = "350 rue des Lilas Ouest Québec Ontario G1L 1B6"
        self.raw_address_diff_PostalCode = "350 rue des Lilas Ouest Québec Québec G1P 1B6"
        self.raw_address_diff_Orientation = "350 rue des Lilas Est Québec Québec G1L 1B6"

        # Ah tu vois c'est ici que l'on doit ce détacher de AddressComparer
        # tu peux aussi juste affecter aux raw_equivalent_comparison etc juste la valeur de retour
        # sinon ça rend le test lourd

        # self.address_comparer = AddressesComparer(self.address_parser_bpemb_device_0)
        first_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        first_address_parsing = [
            ('350', ('StreetNumber', 1.0)),
            ('rue', ('StreetName', 0.9987)),
            ('des', ('StreetName', 0.9993)),
            ('Lilas', ('StreetName', 0.8176)),
            ('Ouest', ('Orientation', 0.781)),
            ('Quebec', ('Municipality', 0.9768)),
            ('Quebec', ('Province', 1.0)),
            ('G1L', ('PostalCode', 0.9993)),
            ('1B6', ('PostalCode', 1.0))]

        first_formatted_parsed_address = FormattedParsedAddress({first_address: first_address_parsing})

        self.raw_equivalent_comparison =
        self.raw_address_diff_streetNumber_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_streetNumber))
        self.raw_address_diff_streetName_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_streetName))
        self.raw_address_diff_Unit_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_Unit))
        self.raw_address_diff_Municipality_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_Municipality))
        self.raw_address_diff_Province_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_Province))
        self.raw_address_diff_PostalCode_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_PostalCode))
        self.raw_address_diff_Orientation_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_Orientation))

    def setup_address_comparer_mock(self, address_one, address_two, model_type="BPEMB"):
        address_parser_mock = MagicMock()
        address_parser_mock.__call__.return_value = [address_one, address_two]
        address_parser_mock.model_type.capitalize.return_value = model_type

        return address_parser_mock

    def test_givenIdenticalRawAddresses_whenCompareRaw_thenReturnIdentical(self):
        raw_address_original = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        raw_address_identical = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        raw_identical_comparison = self.address_comparer.compare_raw(
            (raw_address_original, raw_address_identical))
        self.assertTrue(raw_identical_comparison.indentical)

    def test_givenIdenticalRawAddresses_whenCompareRaw_thenReturnIdentical(self):
        address_parser_mock = self.setup_address_comparer_mock(address_one, address_two)

        address_comparer = AddressesComparer(address_parser_mock)
        expected = ""
        actual = address_comparer.compare_raw((address_one, address_two))
        self.assertEqual(expected, actual)

    def test_identical_raw_address_identical_comparison(self):
        self.raw_address_original = "350 rue des Lilas Ouest Québec Québec G1L 1B6"

        self.assertTrue(self.raw_identical_comparison.indentical)

    def test_identical_raw_address_equivalent_comparison(self):
        self.assertTrue(self.raw_identical_comparison.equivalent)

    def test_equivalent_raw_address_identical_comparison(self):
        self.assertFalse(self.raw_equivalent_comparison.indentical)

    def test_equivalent_raw_address_equivalent_comparison(self):
        self.raw_equivalent_comparison.comparison_report()
        self.assertTrue(self.raw_equivalent_comparison.equivalent)

    def test_streetNumber_diff_raw_address_identical_comparison(self):
        self.assertFalse(self.raw_address_diff_streetNumber_comparison.indentical)

    def test_streetNumber_diff_raw_address_equivalent_comparison(self):
        self.assertFalse(self.raw_address_diff_streetNumber_comparison.equivalent)

    # des tests pour model type


class AddressComparisonOutputTests(CaptureOutputTestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.raw_address_original = "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"
        self.raw_address_identical = "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"
        self.raw_address_equivalent = "350  rue des Lilas Ouest Quebec city Quebec G1L 1B6"
        self.raw_address_diff_streetNumber = "450 rue des Lilas Ouest Quebec city Quebec G1L 1B6"

        self.address_parser = AddressParser(model_type="bpemb", device=1)
        self.address_comparer = AddressesComparer(self.address_parser)

    def test_raw_identical_comparison_report_print_output(self):
        raw_compare_identical = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_identical))
        self._capture_output()
        raw_compare_identical.comparison_report()
        expected = """=============================================================================================================================
Comparison report of the two raw addresses: Identical

Address one: 350 rue des Lilas Ouest Quebec city Quebec G1L 1B6
and
Address two: 350 rue des Lilas Ouest Quebec city Quebec G1L 1B6


Probabilities of parsed tags for the addresses with deepparse using Bpemb: 

Parsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Municipality='Ouest Quebec city', Province='Quebec', PostalCode='G1L 1B6'>
[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('city', ('Municipality', 0.6637)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]
============================================================================================================================="""
        actual = self.test_out.getvalue().strip()
        self.assertEqual(expected, actual)
        # self.assertNotIn(#Code couleur vert et rouge)

    def test_raw_equivalent_comparison_report_print_output(self):
        raw_compare_identical = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_equivalent))
        self._capture_output()
        raw_compare_identical.comparison_report()
        expected = """=============================================================================================================================
Comparison report of the two raw addresses: Equivalent

Address one: 350 rue des Lilas Ouest Quebec city Quebec G1L 1B6
and
Address two: 350  rue des Lilas Ouest Quebec city Quebec G1L 1B6


Probabilities of parsed tags for the addresses with deepparse using Bpemb: 

Parsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Municipality='Ouest Quebec city', Province='Quebec', PostalCode='G1L 1B6'>
[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('city', ('Municipality', 0.6637)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]

Parsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Municipality='Ouest Quebec city', Province='Quebec', PostalCode='G1L 1B6'>
[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('city', ('Municipality', 0.6637)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]


Raw differences between the two addresses: 
White: Shared
Red: Belongs only to Address one
Green: Belongs only to Address two

350  rue des Lilas Ouest Quebec city Quebec G1L 1B6
============================================================================================================================="""
        actual = self.test_out.getvalue().strip()
        self.assertEqual(expected, actual)
        # self.assertIn(#Code couleur vert et rouge)

    def test_raw_not_equivalent_diff_streetNumber_comparison_report_print_output(self):
        raw_compare_identical = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_diff_streetNumber))
        self._capture_output()
        raw_compare_identical.comparison_report()
        expected = """=============================================================================================================================
Comparison report of the two raw addresses: Not equivalent

Address one: 350 rue des Lilas Ouest Quebec city Quebec G1L 1B6
and
Address two: 450 rue des Lilas Ouest Quebec city Quebec G1L 1B6


Probabilities of parsed tags for the addresses with deepparse using Bpemb: 

Parsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Municipality='Ouest Quebec city', Province='Quebec', PostalCode='G1L 1B6'>
[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('city', ('Municipality', 0.6637)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]

Parsed address: FormattedParsedAddress<StreetNumber='450', StreetName='rue des Lilas', Municipality='Ouest Quebec city', Province='Quebec', PostalCode='G1L 1B6'>
[('450', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('city', ('Municipality', 0.6637)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]


Addresses tags differences between the two addresses: 
White: Shared
Red: Belongs only to Address one
Green: Belongs only to Address two

StreetNumber: 
3450
============================================================================================================================="""
        actual = self.test_out.getvalue().strip()
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()


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
        formated_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery"
        ]
