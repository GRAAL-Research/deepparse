# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, protected-access
import unittest
from unittest import TestCase

from deepparse.comparer import FormattedComparedAddresses
from deepparse.parser import FormattedParsedAddress


class AbstractFormattedComparedAddresses(FormattedComparedAddresses):

    def _comparison_report_builder(self):
        return "Comparison report\n"

    def _get_probs(self):
        pass


class TestFormattedComparedAddresses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.a_address_parser_model = "bpemb"

        cls.original_raw_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        cls.original_parsed_address = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                       ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                       ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                       ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                       ('1B6', ('PostalCode', 1.0))]

        cls.original_formatted_parsed_address = FormattedParsedAddress(
            {cls.original_raw_address: cls.original_parsed_address})

    def test_givenIdenticalComparison_thenReturnIdentical(self):
        identical_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        identical_address_parsing = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                     ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                     ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                     ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                     ('1B6', ('PostalCode', 1.0))]

        identical_formatted_parsed_address = FormattedParsedAddress({identical_address: identical_address_parsing})

        raw_identical_comparison = AbstractFormattedComparedAddresses(
            first_address=self.original_formatted_parsed_address,
            second_address=identical_formatted_parsed_address,
            origin=("deepparse using bpemb", "deepparse using bpemb"),
            with_prob=True)

        self.assertTrue(raw_identical_comparison.identical)

    def test_givenEquivalentComparison_thenReturnEquivalent(self):
        equivalent_address = "350  rue des Lilas Ouest Quebec Quebec G1L 1B6"
        equivalent_address_parsing = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                      ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                      ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                      ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                      ('1B6', ('PostalCode', 1.0))]

        equivalent_formatted_address_parsing = FormattedParsedAddress({equivalent_address: equivalent_address_parsing})

        raw_identical_comparison = AbstractFormattedComparedAddresses(
            first_address=self.original_formatted_parsed_address,
            second_address=equivalent_formatted_address_parsing,
            origin=("deepparse using bpemb", "deepparse using bpemb"),
            with_prob=True)

        self.assertTrue(raw_identical_comparison.equivalent)
        self.assertFalse(raw_identical_comparison.identical)

    def test_givenNotEquivalentComparison_thenReturnNotEquivalent(self):
        not_equivalent_address = "450 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        not_equivalent_address_parsing = [('450', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                          ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                          ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                          ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                          ('1B6', ('PostalCode', 1.0))]

        not_equivalent_formatted_address_parsing = FormattedParsedAddress(
            {not_equivalent_address: not_equivalent_address_parsing})

        raw_identical_comparison = AbstractFormattedComparedAddresses(
            first_address=self.original_formatted_parsed_address,
            second_address=not_equivalent_formatted_address_parsing,
            origin=("deepparse using bpemb", "deepparse using bpemb"),
            with_prob=True)

        self.assertFalse(raw_identical_comparison.equivalent)

    def test_comparisonReportSignal(self):
        identical_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        identical_address_parsing = [('350', ('StreetNumber', 1.0)), ('rue', ('StreetName', 0.9987)),
                                     ('des', ('StreetName', 0.9993)), ('Lilas', ('StreetName', 0.8176)),
                                     ('Ouest', ('Orientation', 0.781)), ('Quebec', ('Municipality', 0.9768)),
                                     ('Quebec', ('Province', 1.0)), ('G1L', ('PostalCode', 0.9993)),
                                     ('1B6', ('PostalCode', 1.0))]

        identical_formatted_parsed_address = FormattedParsedAddress({identical_address: identical_address_parsing})

        raw_identical_comparison = AbstractFormattedComparedAddresses(
            first_address=self.original_formatted_parsed_address,
            second_address=identical_formatted_parsed_address,
            origin=("deepparse using bpemb", "deepparse using bpemb"),
            with_prob=True)

        self.assertEqual(raw_identical_comparison._comparison_report(nb_delimiters=20),
                         "====================\nComparison report\n====================\n\n")


if __name__ == "__main__":
    unittest.main()
