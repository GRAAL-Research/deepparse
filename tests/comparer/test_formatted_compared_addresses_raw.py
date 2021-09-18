# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer.formatted_compared_addresses_raw import FormattedComparedAddressesRaw


class FormattedTest():
    def __init__(self, StreetNumber, StreetName, Unit, Municipality, Province, PostalCode, Orientation, GeneralDelivery, raw_adress, address_parsed_components):
        self.StreetNumber = StreetNumber
        self.StreetName = StreetName
        self.Unit = Unit
        self.Municipality = Municipality
        self.Province = Province
        self.PostalCode = PostalCode
        self.Orientation = Orientation
        self.GeneralDelivery = GeneralDelivery
        self.raw_address =  raw_adress
        self.address_parsed_components = address_parsed_components

class FormattedTestIdentical(FormattedTest):
    def to_list_of_tuples(self):
        return [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), ('Ouest', 'Orientation'), (None, 'GeneralDelivery')]

    def __repr__(self):
        return "FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>"

class FormattedTestEquivalent(FormattedTest):
    def to_list_of_tuples(self):
        return [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), ('Ouest', 'Orientation'), (None, 'GeneralDelivery')]

    def __repr__(self):
        return "FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>"


class FormattedTestNotIdentical(FormattedTest):

    def to_list_of_tuples(self):
        return [('450', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Ouest', 'Orientation'), ('Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'GeneralDelivery')]

    def __repr__(self):
        return "FormattedParsedAddress<StreetNumber='450', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>"



class TestFormattedComparedAdressesesRaw(TestCase):


    def test_givenIdenticalAddressesRaw_whenCompareRaw_thenReturnIdenticalComparison_report(self):
        self.maxDiff = None

        FormattedParsedAddressOneIdentical = FormattedTestIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress="350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9987)),
                                        ('des', ('StreetName', 0.9993)),
                                        ('Lilas', ('StreetName', 0.8176)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9768)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9993)),
                                        ('1B6', ('PostalCode', 1.0))])

        

        FormattedParsedAddressTwoIdentical = FormattedTestIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress= "350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9987)),
                                        ('des', ('StreetName', 0.9993)),
                                        ('Lilas', ('StreetName', 0.8176)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9768)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9993)),
                                        ('1B6', ('PostalCode', 1.0))])

        formatted_identical_dict = {'address_one': FormattedParsedAddressOneIdentical,
                                    'address_two':FormattedParsedAddressTwoIdentical,
                                    'colorblind': False,
                                    'origin': ('deepparse using Bpemb',
                                            'deepparse using Bpemb')
}

        formatted_compared_addresses_raw = FormattedComparedAddressesRaw(**formatted_identical_dict)



        self.assertEqual("Comparison report of the two raw addresses: Identical\n\nAddress one: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nAddress two: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n", formatted_compared_addresses_raw._comparison_report_builder())


    def test_givenEquivalentAddressesRaw_whenCompareRaw_thenReturnEquivalentComparison_report(self):
        self.maxDiff = None

        FormattedParsedAddressOneEquivalent = FormattedTestIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress="350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9987)),
                                        ('des', ('StreetName', 0.9993)),
                                        ('Lilas', ('StreetName', 0.8176)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9768)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9993)),
                                        ('1B6', ('PostalCode', 1.0))])

        

        FormattedParsedAddressTwoEquivalent = FormattedTestEquivalent(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress= "350  rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9987)),
                                        ('des', ('StreetName', 0.9993)),
                                        ('Lilas', ('StreetName', 0.8176)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9768)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9993)),
                                        ('1B6', ('PostalCode', 1.0))])

        formatted_equivalent_dict = {'address_one': FormattedParsedAddressOneEquivalent,
                                    'address_two':FormattedParsedAddressTwoEquivalent,
                                    'colorblind': False,
                                    'origin': ('deepparse using Bpemb',
                                            'deepparse using Bpemb')
}

        formatted_compared_addresses_raw = FormattedComparedAddressesRaw(**formatted_equivalent_dict)



        self.assertEqual("Comparison report of the two raw addresses: Equivalent\n\nAddress one: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nAddress two: 350  rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\n\nRaw differences between the two addresses: \nWhite: Shared\nRed: Belongs only to address one\nGreen: Belongs only to address two\n\n\x1b[38;2;255;255;255m350\x1b[0m\x1b[48;2;0;255;0m \x1b[0m\x1b[38;2;255;255;255m rue des Lilas Ouest Quebec Quebec G1L 1B6\x1b[0m\n", formatted_compared_addresses_raw._comparison_report_builder())

#"Comparison report of the two raw addresses: Equivalent\n\nAddress one: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nAddress two: 350  rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', 0.9768, Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', 0.9768, Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Municipality', 0.4356)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\n\nRaw differences between the two addresses: \nWhite: Shared\nRed: Belongs only to address one\nGreen: Belongs only to address two\n\n\x1b[38;2;255;255;255m350\x1b[0m\x1b[48;2;0;255;0m \x1b[0m\x1b[38;2;255;255;255m rue des Lilas Ouest Quebec Quebec G1L 1B6\x1b[0m\n"



    def test_givenDifferentAddressesRaw_whenCompareRaw_thenReturnDifferentComparison_report(self):
        self.maxDiff = None

        FormattedParsedAddressOneDiff = FormattedTestIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress="350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9987)),
                                        ('des', ('StreetName', 0.9993)),
                                        ('Lilas', ('StreetName', 0.8176)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9768)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9993)),
                                        ('1B6', ('PostalCode', 1.0))])


        FormattedParsedAddressTwoDiff = FormattedTestNotIdentical(StreetNumber = '450',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress= "450 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('450', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9987)),
                                        ('des', ('StreetName', 0.9993)),
                                        ('Lilas', ('StreetName', 0.8176)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9768)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9993)),
                                        ('1B6', ('PostalCode', 1.0))])

        formatted_different_dict = {'address_one': FormattedParsedAddressOneDiff,
                                    'address_two':FormattedParsedAddressTwoDiff,
                                    'colorblind': False,
                                    'origin': ('deepparse using Bpemb',
                                            'deepparse using Bpemb')
}

        formatted_compared_addresses_tags = FormattedComparedAddressesRaw(**formatted_different_dict)



        self.assertEqual("Comparison report of the two raw addresses: Not equivalent\n\nAddress one: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\nand\nAddress two: 450 rue des Lilas Ouest Quebec Quebec G1L 1B6\n\n\nProbabilities of parsed tags for the addresses with deepparse using Bpemb: \n\nParsed address: FormattedParsedAddress<StreetNumber='350', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\nParsed address: FormattedParsedAddress<StreetNumber='450', StreetName='rue des Lilas', Orientation='Ouest', Municipality='Quebec', Province='Quebec', PostalCode='G1L 1B6'>\n[('450', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)), ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)), ('1B6', ('PostalCode', 1.0))]\n\n\nAddresses tags differences between the two addresses: \nWhite: Shared\nRed: Belongs only to address one\nGreen: Belongs only to address two\n\nStreetNumber: \n\x1b[38;2;255;0;0m3\x1b[0m\x1b[38;2;0;255;0m4\x1b[0m\x1b[38;2;255;255;255m50\x1b[0m\n"
, formatted_compared_addresses_tags._comparison_report_builder())



if __name__ == "__main__":
    unittest.main()


