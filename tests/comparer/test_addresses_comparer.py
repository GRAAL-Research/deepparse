# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from deepparse.comparer.addresses_comparer import AddressesComparer
from deepparse.comparer.formatted_compared_addresses_raw import FormattedComparedAddressesRaw
from deepparse.comparer.formatted_compared_addresses_tags import FormattedComparedAddressesTags


class TestAdressesComparer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_type = "bpemb"

        cls.raw_address_original = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        cls.raw_address_original_parsed = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                           ("Ouest", "Orientation"), ("Québec", "Municipality"), ("Québec", "Province"),
                                           ("G1L 1B6", "PostalCode")]

        cls.tagged_addresses_components_with_prob = [('305', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9989)),
                                                     ('des', ('StreetName', 0.9998)), ('Lilas', ('StreetName', 0.9343)),
                                                     ('Ouest', ('Municipality', 0.781)),
                                                     ('Québec', ('Municipality', 0.9467)),
                                                     ('Québec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9997)),
                                                     ('1B6', ('PostalCode', 1.0))]
        cls.tagged_addresses_components = [('305', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'),
                                           ('Ouest Québec', 'Municipality'), ('Québec', 'Province'),
                                           ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')]

    def setup_address_parser_mock(self, comparison_list, model_type="bpemb"):
        address_parser_mock = MagicMock()

        address_parser_mock.__call__().return_value = comparison_list
        address_parser_mock.model_type.capitalize().return_value = model_type

        return address_parser_mock

    def test_givenARawAddress_whenCompareRaw_thenReturnProperClass(self):
        raw_address_identical = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"

        raw_address_identical_parsed = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                        ("Ouest", "Orientation"), ("Québec", "Municipality"), ("Québec", "Province"),
                                        ("G1L 1B6", "PostalCode")]

        mocked_parser = self.setup_address_parser_mock([self.raw_address_original_parsed, raw_address_identical_parsed])
        address_comparer = AddressesComparer(mocked_parser)
        raw_one_comparison = address_comparer.compare_raw((self.raw_address_original, raw_address_identical))
        self.assertIsInstance(raw_one_comparison, FormattedComparedAddressesRaw)

    def test_givenMultipleRawAddresses_whenCompareRaw_thenReturnProperClass(self):
        raw_address_identical = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"

        raw_address_identical_parsed = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                        ("Ouest", "Orientation"), ("Québec", "Municipality"), ("Québec", "Province"),
                                        ("G1L 1B6", "PostalCode")]

        raw_address_diff_StreetNumber = "450 rue des Lilas Ouest Quebec Quebec G1L 1B6"

        raw_address_diff_StreetNumber_parsed = [("450", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                                ("Ouest", "Orientation"), ("Québec", "Municipality"),
                                                ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

        mocked_parser = self.setup_address_parser_mock(
            [[self.raw_address_original_parsed, raw_address_identical_parsed],
             [self.raw_address_original_parsed, raw_address_diff_StreetNumber_parsed]])
        address_comparer = AddressesComparer(mocked_parser)

        raw_multiple_comparisons = address_comparer.compare_raw([(self.raw_address_original, raw_address_identical),
                                                                 (self.raw_address_original,
                                                                  raw_address_diff_StreetNumber)])
        self.assertIsInstance(raw_multiple_comparisons, list)
        self.assertIsInstance(raw_multiple_comparisons[0], FormattedComparedAddressesRaw)
        self.assertIsInstance(raw_multiple_comparisons[1], FormattedComparedAddressesRaw)

    def test_givenATaggedAddress_whenCompareTags_thenReturnProperClass(self):
        mocked_parser = self.setup_address_parser_mock(self.raw_address_original_parsed)
        address_comparer = AddressesComparer(mocked_parser)

        tags_one_comparison = address_comparer.compare_tags(self.raw_address_original_parsed)
        self.assertIsInstance(tags_one_comparison, FormattedComparedAddressesTags)

    def test_givenMultipleTaggedAddresses_whenCompareTags_thenReturnProperClass(self):
        raw_address_diff_StreetNumber_parsed = [("450", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                                ("Ouest", "Orientation"), ("Québec", "Municipality"),
                                                ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

        mocked_parser = self.setup_address_parser_mock(
            [self.raw_address_original_parsed, raw_address_diff_StreetNumber_parsed])
        address_comparer = AddressesComparer(mocked_parser)

        tags_multiple_comparisons = address_comparer.compare_tags(
            [self.raw_address_original_parsed, raw_address_diff_StreetNumber_parsed])
        self.assertIsInstance(tags_multiple_comparisons, list)
        self.assertIsInstance(tags_multiple_comparisons[0], FormattedComparedAddressesTags)
        self.assertIsInstance(tags_multiple_comparisons[1], FormattedComparedAddressesTags)


if __name__ == "__main__":
    unittest.main()
