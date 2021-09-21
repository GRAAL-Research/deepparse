# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer.formatted_compared_addresses_raw import FormattedComparedAddressesRaw
from deepparse.parser import FormattedParsedAddress

# Ici, nous n'avons pas besoin de mock, car on fait juste créer le parsing manuellement.
class TestFormattedComparedAddressesRaw(TestCase):

    def test_givenIdenticalAddressesRaw_whenCompareRaw_thenReturnIdenticalComparison_report(self):
        self.maxDiff = None

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

        second_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        second_address_parsing = [
            ('350', ('StreetNumber', 1.0)),
            ('rue', ('StreetName', 0.9987)),
            ('des', ('StreetName', 0.9993)),
            ('Lilas', ('StreetName', 0.8176)),
            ('Ouest', ('Orientation', 0.781)),
            ('Quebec', ('Municipality', 0.9768)),
            ('Quebec', ('Province', 1.0)),
            ('G1L', ('PostalCode', 0.9993)),
            ('1B6', ('PostalCode', 1.0))]

        second_formatted_parsed_address = FormattedParsedAddress({second_address: second_address_parsing})

        formatted_identical_dict = {'address_one': first_formatted_parsed_address,
                                    'address_two': second_formatted_parsed_address,
                                    'colorblind': False,
                                    'origin': ('deepparse using Bpemb',
                                               'deepparse using Bpemb')
                                    }

        formatted_compared_addresses_raw = FormattedComparedAddressesRaw(**formatted_identical_dict)
        expected = "Comparison report of the two raw addresses: Identical\n\nAddress one: 350 rue des Lilas Ouest " \
                   "Quebec Quebec G1L 1B6\nand\nAddress two: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\n" \
                   "Probabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: " \
                   "FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', " \
                   "Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0))," \
                   " ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', " \
                   "0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), " \
                   "('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n"

        actual = formatted_compared_addresses_raw._comparison_report_builder()

        self.assertEqual(expected, actual)

    def test_givenEquivalentAddressesRaw_whenCompareRaw_thenReturnEquivalentComparisonReport(self):
        self.maxDiff = None

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

        second_address = "350  rue des Lilas Ouest Quebec Quebec G1L 1B6"  # not identical address with the preceding
        second_address_parsing = [
            ('350', ('StreetNumber', 1.0)),
            ('rue', ('StreetName', 0.9987)),
            ('des', ('StreetName', 0.9993)),
            ('Lilas', ('StreetName', 0.8176)),
            ('Ouest', ('Orientation', 0.781)),
            ('Quebec', ('Municipality', 0.9768)),
            ('Quebec', ('Province', 1.0)),
            ('G1L', ('PostalCode', 0.9993)),
            ('1B6', ('PostalCode', 1.0))]

        second_formatted_parsed_address = FormattedParsedAddress({second_address: second_address_parsing})

        formatted_identical_dict = {'address_one': first_formatted_parsed_address,
                                    'address_two': second_formatted_parsed_address,
                                    'colorblind': False,
                                    'origin': ('deepparse using Bpemb',
                                               'deepparse using Bpemb')
                                    }

        formatted_compared_addresses_raw = FormattedComparedAddressesRaw(**formatted_identical_dict)

        expected = "Comparison report of the two raw addresses: Equivalent\n\nAddress one: 350 rue des Lilas Ouest " \
                   "Quebec Quebec G1L 1B6\nand\nAddress two: 350  rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\n" \
                   "Probabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: " \
                   "FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', " \
                   "Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', " \
                   "('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), " \
                   "('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', " \
                   "('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), " \
                   "('1B6', ('PostalCode', 1.0))]\n\nParsed address: FormattedParsedAddress<StreetNumber='350', " \
                   "StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', " \
                   "PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), " \
                   "('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', " \
                   "0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', " \
                   "('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\n\nRaw differences between the two a" \
                   "ddresses: \nWhite: Shared\nRed: Belongs only to address one\nGreen: Belongs only to address " \
                   "two\n\n\x1b[38;2;255;255;255m350\x1b[0m\x1b[48;2;0;255;0m \x1b[0m\x1b[38;2;255;255;255m " \
                   "rue des Lilas Ouest Quebec Quebec G1L 1B6\x1b[0m\n"

        actual = formatted_compared_addresses_raw._comparison_report_builder()
        self.assertEqual(expected, actual)

    #todo to be fix
    def test_givenDifferentAddressesRaw_whenCompareRaw_thenReturnDifferentComparison_report(self):
        self.maxDiff = None

        FormattedParsedAddressOneDiff = FormattedTestIdentical(StreetNumber='350',
                                                               StreetName='rue des Lilas',
                                                               Unit=None,
                                                               Municipality='Quebec',
                                                               Province='Quebec',
                                                               PostalCode='G1L 1B6',
                                                               Orientation='Ouest',
                                                               GeneralDelivery=None,
                                                               raw_adress="350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                                               address_parsed_components=[
                                                                   ('350', ('StreetNumber', 1.0)),
                                                                   ('rue', ('StreetName', 0.9987)),
                                                                   ('des', ('StreetName', 0.9993)),
                                                                   ('Lilas', ('StreetName', 0.8176)),
                                                                   ('Ouest', ('Orientation', 0.781)),
                                                                   ('Quebec', ('Municipality', 0.9768)),
                                                                   ('Quebec', ('Province', 1.0)),
                                                                   ('G1L', ('PostalCode', 0.9993)),
                                                                   ('1B6', ('PostalCode', 1.0))])

        FormattedParsedAddressTwoDiff = FormattedTestNotIdentical(StreetNumber='450',
                                                                  StreetName='rue des Lilas',
                                                                  Unit=None,
                                                                  Municipality='Quebec',
                                                                  Province='Quebec',
                                                                  PostalCode='G1L 1B6',
                                                                  Orientation='Ouest',
                                                                  GeneralDelivery=None,
                                                                  raw_adress="450 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                                                  address_parsed_components=[
                                                                      ('450', ('StreetNumber', 1.0)),
                                                                      ('rue', ('StreetName', 0.9987)),
                                                                      ('des', ('StreetName', 0.9993)),
                                                                      ('Lilas', ('StreetName', 0.8176)),
                                                                      ('Ouest', ('Orientation', 0.781)),
                                                                      ('Quebec', ('Municipality', 0.9768)),
                                                                      ('Quebec', ('Province', 1.0)),
                                                                      ('G1L', ('PostalCode', 0.9993)),
                                                                      ('1B6', ('PostalCode', 1.0))])

        formatted_different_dict = {'address_one': FormattedParsedAddressOneDiff,
                                    'address_two': FormattedParsedAddressTwoDiff,
                                    'colorblind': False,
                                    'origin': ('deepparse using Bpemb',
                                               'deepparse using Bpemb')
                                    }

        formatted_compared_addresses_tags = FormattedComparedAddressesRaw(**formatted_different_dict)

        self.assertEqual(
            "Comparison report of the two raw addresses: Not equivalent\n\nAddress one: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nAddress two: 450 rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\nParsed address: FormattedParsedAddress<StreetNumber='450', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('450', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\n\nAddresses tags differences between the two addresses: \nWhite: Shared\nRed: Belongs only to address one\nGreen: Belongs only to address two\n\nStreetNumber: \n\x1b[38;2;255;0;0m3\x1b[0m\x1b[38;2;0;255;0m4\x1b[0m\x1b[38;2;255;255;255m50\x1b[0m\n"
            , formatted_compared_addresses_tags._comparison_report_builder())


if __name__ == "__main__":
    unittest.main()
