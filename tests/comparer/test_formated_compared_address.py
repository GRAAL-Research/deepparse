# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer import AdressComparer
from deepparse.parser.address_parser import AddressParser

class TestAdressComparer(TestCase):

        
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


        self.address_parser_bpemb_device_0 = AddressParser(model_type="bpemb", device=0)
        self.address_comparer = AdressComparer(self.address_parser_bpemb_device_0)

        self.raw_identical_comparison = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_identical))
        self.raw_equivalent_comparison = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_equivalent))
        self.raw_address_diff_streetNumber_comparison = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_streetNumber))
        self.raw_address_diff_streetName_comparison  = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_streetName))
        self.raw_address_diff_Unit_comparison  = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_Unit))
        self.raw_address_diff_Municipality_comparison  = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_Municipality))
        self.raw_address_diff_Province_comparison  = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_Province))
        self.raw_address_diff_PostalCode_comparison  = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_PostalCode))
        self.raw_address_diff_Orientation_comparison  = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_diff_Orientation))


    def test_identical_raw_address_identical_comparison(self):
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
    


    #def test_sameAddress_emptyDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_same), {})

    #def test_diff_streetNumberDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_streetNumber), {'StreetNumber': {'base': '350', 'compared': '450'}})

    #def test_diff_streetNameDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_streetName), {'StreetName': {'base': 'rue des Lilas', 'compared': 'Boulevard des Lilas'}})

    #def test_diff_UnitDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one == self.parsed_address_diff_Unit, )

    #def test_diff_MunicipalityDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one == self.parsed_address_diff_Municipality)
        
    #def test_diff_ProvinceDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_Province), {'Province': {'base': 'Québec', 'compared': 'Ontario'}})
        
    #def test_diff_PostalCodeDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one.delta_dict(self.parsed_address_diff_PostalCode), {'PostalCode': {'base': 'G1L 1B6', 'compared': 'G1P 1B6'}})
        
    #def test_diff_OrientationDeltaDict(self):
    #    self.assertEqual(self.parsed_address_one == self.parsed_address_diff_Orientation)


if __name__ == "__main__":
    unittest.main()
