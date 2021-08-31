# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer import AdressComparer
from deepparse.comparer import FormatedComparedAddress
from deepparse.parser.address_parser import AddressParser

class TestAdressComparer(TestCase):

        
    def setUp(self) -> None:
        self.list_of_tuples_address_original = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                    ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

        self.list_of_tuples_address_diff_StreetNumber = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                    ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

        self.raw_address_original = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_identical = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_diff_StreetNumber = "355 rue des Lilas Ouest Québec Québec G1L 1B6"


        self.address_parser_bpemb_device_0 = AddressParser(model_type="bpemb", device=0)
        self.address_comparer = AdressComparer(self.address_parser_bpemb_device_0)

        self.raw_one_comparison = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_identical))
        self.raw_multiple_comparisons = self.address_comparer.compare_raw([(self.raw_address_original, self.raw_address_identical),
                                                                            (self.raw_address_original, self.raw_address_diff_StreetNumber)])

        self.tags_one_comparison = self.address_comparer.compare_tags(self.list_of_tuples_address_original)
        self.tags_multiple_comparisons = self.address_comparer.compare_tags([self.list_of_tuples_address_original, self.list_of_tuples_address_diff_StreetNumber])



        

    def test_raw_one_comparison(self):
        self.assertIsInstance(self.raw_one_comparison, FormatedComparedAddress)
    
    def test_raw_multiple_comparisons(self):
        self.assertIsInstance(self.raw_multiple_comparisons, list)
        self.assertIsInstance(self.raw_multiple_comparisons[0], FormatedComparedAddress)
        self.assertIsInstance(self.raw_multiple_comparisons[1], FormatedComparedAddress)


    def test_tags_one_comparison(self):
        self.assertIsInstance(self.tags_one_comparison, FormatedComparedAddress)
    
    def test_tags_multiple_comparisons(self):
        self.assertIsInstance(self.tags_multiple_comparisons, list)
        self.assertIsInstance(self.tags_multiple_comparisons[0], FormatedComparedAddress)
        self.assertIsInstance(self.tags_multiple_comparisons[1], FormatedComparedAddress)



if __name__ == "__main__":
    unittest.main()
