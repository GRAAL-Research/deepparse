# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

from deepparse.comparer.addresses_comparer import AddressesComparer
from deepparse.comparer.formatted_compared_addresses_raw import FormattedComparedAddressesRaw
from deepparse.comparer.formatted_compared_addresses_tags import FormattedComparedAddressesTags
from deepparse.parser.address_parser import AddressParser


class TestAdressesComparer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_type = "bpemb"
        cls.tagged_addresses_components_with_prob = [('305', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9989)),
                                                     ('des', ('StreetName', 0.9998)), ('Lilas', ('StreetName', 0.9343)),
                                                     ('Ouest', ('Municipality', 0.781)),
                                                     ('Québec', ('Municipality', 0.9467)),
                                                     ('Québec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9997)),
                                                     ('1B6', ('PostalCode', 1.0))]
        cls.tagged_addresses_components = [('305', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'),
                                           ('Ouest Québec', 'Municipality'), ('Québec', 'Province'),
                                           ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')]

    def setUp(self) -> None:
        self.list_of_tuples_address_original = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                                ("Ouest", "Orientation"),
                                                ("Québec", "Municipality"), ("Québec", "Province"),
                                                ("G1L 1B6", "PostalCode")]

        self.list_of_tuples_address_diff_StreetNumber = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                                         ("Ouest", "Orientation"),
                                                         ("Québec", "Municipality"), ("Québec", "Province"),
                                                         ("G1L 1B6", "PostalCode")]

        self.raw_address_original = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_identical = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
        self.raw_address_diff_StreetNumber = "355 rue des Lilas Ouest Québec Québec G1L 1B6"

        self.address_parser_bpemb_device_0 = AddressParser(model_type="bpemb", device=0)
        self.address_comparer = AddressesComparer(self.address_parser_bpemb_device_0)

        self.raw_one_comparison = self.address_comparer.compare_raw(
            (self.raw_address_original, self.raw_address_identical))
        self.raw_multiple_comparisons = self.address_comparer.compare_raw(
            [(self.raw_address_original, self.raw_address_identical),
             (self.raw_address_original, self.raw_address_diff_StreetNumber)])

        self.tags_one_comparison = self.address_comparer.compare_tags(self.list_of_tuples_address_original)
        self.tags_multiple_comparisons = self.address_comparer.compare_tags(
            [self.list_of_tuples_address_original, self.list_of_tuples_address_diff_StreetNumber])

    # la nomenclature du nom d'un test c'est comme ça
    # def test_givenARawAddress_whenCompareRaw_thenReturnProperClass
    # Et ça: self.address_comparer.compare_raw((self.raw_address_original, self.raw_address_identical))
    # ça doit être fait dans le test c'est le setup du test.
    def test_raw_one_comparison(self):
        self.assertIsInstance(self.raw_one_comparison, FormattedComparedAddressesRaw)

    def test_raw_multiple_comparisons(self):
        self.assertIsInstance(self.raw_multiple_comparisons, list)
        self.assertIsInstance(self.raw_multiple_comparisons[0], FormattedComparedAddressesRaw)
        self.assertIsInstance(self.raw_multiple_comparisons[1], FormattedComparedAddressesRaw)

    # il va manquer aussi des tests à savoir si le formatted_comparisons est correct.

    def test_tags_one_comparison(self):
        self.assertIsInstance(self.tags_one_comparison, FormattedComparedAddressesTags)

    def test_tags_multiple_comparisons(self):
        self.assertIsInstance(self.tags_multiple_comparisons, list)
        self.assertIsInstance(self.tags_multiple_comparisons[0], FormattedComparedAddressesTags)
        self.assertIsInstance(self.tags_multiple_comparisons[1], FormattedComparedAddressesTags)

    # il va manquer aussi des tests à savoir si le formatted_comparisons est correct.


if __name__ == "__main__":
    unittest.main()
