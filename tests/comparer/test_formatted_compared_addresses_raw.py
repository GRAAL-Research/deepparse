# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, protected-access
import unittest
from unittest import TestCase

from deepparse.comparer.formatted_compared_addresses_raw import FormattedComparedAddressesRaw
from deepparse.parser import FormattedParsedAddress


class TestFormattedComparedAddressesRaw(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

        cls.original_raw_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        cls.original_parsed_address = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                       ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                       ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                       ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                       ('1B6', ('PostalCode', 1.0))]

        cls.original_formatted_parsed_address = FormattedParsedAddress(
            {cls.original_raw_address: cls.original_parsed_address})

    def test_givenIdenticalAddressesRaw_whenCompareRaw_thenReturnIdenticalComparisonReport(self):
        identical_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        identical_address_parsing = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                     ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                     ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                     ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                     ('1B6', ('PostalCode', 1.0))]

        identical_formatted_parsed_address = FormattedParsedAddress({identical_address: identical_address_parsing})

        identical_formatted_compared_addresses_raw = FormattedComparedAddressesRaw(
            first_address=self.original_formatted_parsed_address,
            second_address=identical_formatted_parsed_address,
            origin=('deepparse using Bpemb', 'deepparse using Bpemb'),
            with_prob=True)

        expected = "Comparison report of the two raw addresses: Identical\n\nAddress : 350 rue des Lilas Ouest " \
                   "Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses with deepparse using " \
                   "Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas'" \
                   ", Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', " \
                   "('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), " \
                   "('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', " \
                   "('Municipality', 0.9768)), ('Quebec', ('Province', 1.0))" \
                   ", ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n"

        actual = identical_formatted_compared_addresses_raw._comparison_report_builder()

        self.assertEqual(expected, actual)

    def test_givenEquivalentAddressesRaw_whenCompareRaw_thenReturnEquivalentComparisonReport(self):
        # Not identical address with the preceding
        equivalent_address = "350  rue des Lilas Ouest Quebec Quebec G1L 1B6"
        equivalent_address_parsing = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                      ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                      ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                      ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                      ('1B6', ('PostalCode', 1.0))]

        equivalent_formatted_parsed_address = FormattedParsedAddress({equivalent_address: equivalent_address_parsing})

        equivalent_formatted_compared_addresses_raw_ = FormattedComparedAddressesRaw(
            first_address=self.original_formatted_parsed_address,
            second_address=equivalent_formatted_parsed_address,
            origin=('deepparse using Bpemb', 'deepparse using Bpemb'),
            with_prob=True)

        expected = "Comparison report of the two raw addresses: Equivalent\n\nFirst address " \
                   ": 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nSecond address: 350  rue des" \
                   " Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses" \
                   " with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350'," \
                   " StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', " \
                   "PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), " \
                   "('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', " \
                   "('Orientation', 0.781)), ('Quebec'," \
                   " ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993))" \
                   ", ('1B6', ('PostalCode', 1.0))]\n\nParsed address: FormattedParsedAddress<StreetNumber='350', " \
                   "StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', " \
                   "PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), " \
                   "('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', " \
                   "('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768))," \
                   " ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), " \
                   "('1B6', ('PostalCode', 1.0))]\n\n\nRaw" \
                   " differences between the two addresses: \nWhite: Shared\nBlue: " \
                   "Belongs only to the first address\nYellow:" \
                   " Belongs only to the second address\n\n\x1b[38;2;255;255;255m350" \
                   "\x1b[0m\x1b[48;2;255;194;10m \x1b[0m\x1b" \
                   "[38;2;255;255;255m rue des Lilas Ouest Quebec Quebec G1L 1B6\x1b[0m\n"

        actual = equivalent_formatted_compared_addresses_raw_._comparison_report_builder()

        self.assertEqual(expected, actual)

    def test_givenDifferentAddressesRaw_whenCompareRaw_thenReturnDifferentComparisonReport(self):
        # Not identical address with the preceding
        not_equivalent_address = "450 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        not_equivalent_address_parsing = [('450', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                          ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                          ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                          ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                          ('1B6', ('PostalCode', 1.0))]

        not_equivalent_formatted_parsed_address = FormattedParsedAddress(
            {not_equivalent_address: not_equivalent_address_parsing})

        not_equivalent_formatted_compared_addresses_raw_ = FormattedComparedAddressesRaw(
            first_address=self.original_formatted_parsed_address,
            second_address=not_equivalent_formatted_parsed_address,
            origin=('deepparse using Bpemb', 'deepparse using Bpemb'),
            with_prob=True)

        expected = "Comparison report of the two raw addresses: Not equivalent\n\nFirst address " \
                   ": 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nSecond address: 450 rue des" \
                   " Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses" \
                   " with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350'," \
                   " StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', " \
                   "Province='Quebec', PostalCode='G1L" \
                   " 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName'," \
                   " 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec'," \
                   " ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993))," \
                   " ('1B6', ('PostalCode', 1.0))]\n\nParsed address: " \
                   "FormattedParsedAddress<StreetNumber='450', StreetName" \
                   "='rue des Lilas', Orientation='Ouest', Municipality=" \
                   "'Quebec', Province='Quebec', PostalCode='G1L 1B6'>" \
                   "\n[('450', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987))," \
                   " ('des', ('StreetName', 0.9993))," \
                   " ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781))," \
                   " ('Quebec', ('Municipality', 0.9768))," \
                   " ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993))," \
                   " ('1B6', ('PostalCode', 1.0))]\n\n\nAddresses" \
                   " tags differences between the two addresses: \nWhite:" \
                   " Shared\nBlue: Belongs only to the first address\nYellow:" \
                   " Belongs only to the second address\n\nStreetNumber:" \
                   " \n\x1b[38;2;26;123;220m3\x1b[0m\x1b[38;2;255;194;10m4\x1b[0" \
                   "m\x1b[38;2;255;255;255m50\x1b[0m\n"

        actual = not_equivalent_formatted_compared_addresses_raw_._comparison_report_builder()
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
