# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer import AdressComparer
from deepparse.parser.address_parser import AddressParser

class TestAdressComparer(TestCase):

        
    def setUp(self) -> None:
        self.address_parser = AddressParser(model_type="bpemb", device=0)
        self.parsed_address_one = AdressComparer(self.address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6"))
        self.parsed_address_same = self.address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")
        self.parsed_address_diff_streetNumber = self.address_parser("450 rue des Lilas Ouest Québec Québec G1L 1B6")
        self.parsed_address_diff_streetName = self.address_parser("350 Boulevard des Lilas Ouest Québec Québec G1L 1B6")
        self.parsed_address_diff_Unit = self.address_parser("350 rue des Lilas Ouest app 105 Québec Québec G1L 1B6")
        self.parsed_address_diff_Municipality = self.address_parser("350 rue des Lilas Ouest Ste-Foy Québec G1L 1B6")
        self.parsed_address_diff_Province = self.address_parser("350 rue des Lilas Ouest Québec Ontario G1L 1B6")
        self.parsed_address_diff_PostalCode = self.address_parser("350 rue des Lilas Ouest Québec Québec G1P 1B6")
        self.parsed_address_diff_Orientation = self.address_parser("350 rue des Lilas Est Québec Québec G1L 1B6")


    def test_sameAddress_comparison(self):
        self.assertTrue(self.parsed_address_one == self.parsed_address_same)

    def test_diff_streetNumber_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_streetNumber)

    def test_diff_streetName_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_streetName)

    def test_diff_Unit_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_Unit)

    def test_diff_Municipality_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_Municipality)
        
    def test_diff_Province_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_Province)
        
    def test_diff_PostalCode_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_PostalCode)
        
    def test_diff_Orientation_comparison(self):
        self.assertFalse(self.parsed_address_one == self.parsed_address_diff_Orientation)
    



    def test_sameAddress_emptyDeltaDict(self):
        self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_same), {})

    def test_diff_streetNumberDeltaDict(self):
        self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_streetNumber), {'StreetNumber': {'base': '350', 'compared': '450'}})

    def test_diff_streetNameDeltaDict(self):
        self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_streetName), {'StreetName': {'base': 'rue des Lilas', 'compared': 'Boulevard des Lilas'}})

    #def test_diff_UnitDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one == self.parsed_address_diff_Unit, )

    #def test_diff_MunicipalityDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one == self.parsed_address_diff_Municipality)
        
    def test_diff_ProvinceDeltaDict(self):
        self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_Province), {'Province': {'base': 'Québec', 'compared': 'Ontario'}})
        
    def test_diff_PostalCodeDeltaDict(self):
        self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_PostalCode), {'PostalCode': {'base': 'G1L 1B6', 'compared': 'G1P 1B6'}})
        
    #def test_diff_OrientationDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one == self.parsed_address_diff_Orientation)


if __name__ == "__main__":
    unittest.main()