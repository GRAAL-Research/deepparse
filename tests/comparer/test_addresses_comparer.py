# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer.addresses_comparer import AdressesComparer
from deepparse.comparer.formatted_compared_addresses_raw import FormattedComparedAddressesRaw
from deepparse.comparer.formatted_compared_addresses_tags import FormattedComparedAddressesTags
from deepparse.parser import AddressParser

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
        self.address_comparer = AdressesComparer(self.address_parser_bpemb_device_0)

        self.raw_one_comparison = self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_identical))
        self.raw_multiple_comparisons = self.address_comparer.compare_raw([(self.raw_address_original, self.raw_address_identical),
                                                                            (self.raw_address_original, self.raw_address_diff_StreetNumber)])

        self.tags_one_comparison = self.address_comparer.compare_tags(self.list_of_tuples_address_original)
        self.tags_multiple_comparisons = self.address_comparer.compare_tags([self.list_of_tuples_address_original, self.list_of_tuples_address_diff_StreetNumber])



        

    def test_raw_one_comparison(self):
        self.assertIsInstance(self.raw_one_comparison, FormattedComparedAddressesRaw)
    
    def test_raw_multiple_comparisons(self):
        self.assertIsInstance(self.raw_multiple_comparisons, list)
        self.assertIsInstance(self.raw_multiple_comparisons[0], FormattedComparedAddressesRaw)
        self.assertIsInstance(self.raw_multiple_comparisons[1], FormattedComparedAddressesRaw)


    def test_tags_one_comparison(self):
        self.assertIsInstance(self.tags_one_comparison, FormattedComparedAddressesTags)
    
    def test_tags_multiple_comparisons(self):
        self.assertIsInstance(self.tags_multiple_comparisons, list)
        self.assertIsInstance(self.tags_multiple_comparisons[0], FormattedComparedAddressesTags)
        self.assertIsInstance(self.tags_multiple_comparisons[1], FormattedComparedAddressesTags)



if __name__ == "__main__":
    unittest.main()
