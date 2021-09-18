# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase
from unittest.mock import patch, Mock

from deepparse.comparer.formatted_compared_addresses_tags import FormattedComparedAddressesTags



mock = Mock()
FormattedParsedAddress = mock


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


class FormattedTestNotIdentical(FormattedTest):

    def to_list_of_tuples(self):
        return [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Ouest Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')]




class FormattedComparedAdressesesTagsTest(TestCase):
    


    def test_givenIdenticalAddressesTags_whenCompareTags_thenReturnIdenticalComparison_report(self):
        self.maxDiff = None

        FormattedParsedAddressOne = FormattedTestIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress="350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9989)),
                                        ('des', ('StreetName', 0.9998)),
                                        ('Lilas', ('StreetName', 0.9343)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9467)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9997)),
                                        ('1B6', ('PostalCode', 1.0))])

        

        FormattedParsedAddressTwo = FormattedTestIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = 'Ouest',
                                  GeneralDelivery = None,
                                  raw_adress= "350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9989)),
                                        ('des', ('StreetName', 0.9998)),
                                        ('Lilas', ('StreetName', 0.9343)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9467)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9997)),
                                        ('1B6', ('PostalCode', 1.0))])

        formatted_identical_dict = {'address_one': FormattedParsedAddressOne,
                                    'address_two':FormattedParsedAddressTwo,
                                    'colorblind': False,
                                    'origin': ('source',
                                            'deepparse using Bpemb')
}

        formatted_compared_addresses_tags = FormattedComparedAddressesTags(**formatted_identical_dict)



        self.assertEqual("Comparison report of tags for parsed address: Identical\nRaw address: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\n\nTags: \nsource: [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), ('Ouest', 'Orientation'), (None, 'GeneralDelivery')]\n\ndeepparse using Bpemb: [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), ('Ouest', 'Orientation'), (None, 'GeneralDelivery')]\n\nProbabilities of parsed tags for the address:\n\nRaw address: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9989)), ('des', ('StreetName', 0.9998)), ('Lilas', ('StreetName', 0.9343)), ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9467)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9997)), ('1B6', ('PostalCode', 1.0))]\n", formatted_compared_addresses_tags._comparison_report_builder())




    def test_givenDifferentAddressesTags_whenCompareTags_thenReturnNotIdenticalComparison_report(self):
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
                                        ('rue', ('StreetName', 0.9989)),
                                        ('des', ('StreetName', 0.9998)),
                                        ('Lilas', ('StreetName', 0.9343)),
                                        ('Ouest', ('Orientation', 0.781)),
                                        ('Quebec', ('Municipality', 0.9467)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9997)),
                                        ('1B6', ('PostalCode', 1.0))])


        FormattedParsedAddressTwoDiff = FormattedTestNotIdentical(StreetNumber = '350',
                                  StreetName = 'rue des Lilas',
                                  Unit = None,
                                  Municipality = 'Ouest Quebec',
                                  Province = 'Quebec',
                                  PostalCode = 'G1L 1B6',
                                  Orientation = None,
                                  GeneralDelivery = None,
                                  raw_adress= "350 rue des Lilas Ouest Quebec Quebec G1L 1B6",
                                  address_parsed_components =[('350', ('StreetNumber', 1.0)),
                                        ('rue', ('StreetName', 0.9989)),
                                        ('des', ('StreetName', 0.9998)),
                                        ('Lilas', ('StreetName', 0.9343)),
                                        ('Ouest', ('Municipality', 0.781)),
                                        ('Quebec', ('Municipality', 0.9467)),
                                        ('Quebec', ('Province', 1.0)),
                                        ('G1L', ('PostalCode', 0.9997)),
                                        ('1B6', ('PostalCode', 1.0))])

        formatted_different_dict = {'address_one': FormattedParsedAddressOneDiff,
                                    'address_two':FormattedParsedAddressTwoDiff,
                                    'colorblind': False,
                                    'origin': ('source',
                                            'deepparse using Bpemb')
}

        formatted_compared_addresses_tags = FormattedComparedAddressesTags(**formatted_different_dict)



        self.assertEqual("Comparison report of tags for parsed address: Not identical\nRaw address: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\n\nTags: \nsource: [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), ('Ouest', 'Orientation'), (None, 'GeneralDelivery')]\n\ndeepparse using Bpemb: [('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Ouest Quebec', 'Municipality'), ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')]\n\nProbabilities of parsed tags for the address:\n\nRaw address: 350 rue des Lilas Ouest Quebec Quebec G1L 1B6\n[('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9989)), ('des', ('StreetName', 0.9998)), ('Lilas', ('StreetName', 0.9343)), ('Ouest', ('Municipality', 0.781)), ('Quebec', ('Municipality', 0.9467)), ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9997)), ('1B6', ('PostalCode', 1.0))]\n\n\nAddresses tags differences between the two parsing:\nWhite: Shared\nRed: Belongs only to source\nGreen: Belongs only to deepparse using Bpemb\n\nOrientation: \n\x1b[38;2;255;0;0mOuest\x1b[0m\nMunicipality: \n\x1b[38;2;0;255;0mOuest \x1b[0m\x1b[38;2;255;255;255mQuebec\x1b[0m\n", formatted_compared_addresses_tags._comparison_report_builder())

if __name__ == "__main__":
    unittest.main()



