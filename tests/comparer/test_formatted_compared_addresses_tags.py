# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, protected-access
import unittest
from unittest import TestCase

from deepparse.comparer.formatted_compared_addresses_tags import FormattedComparedAddressesTags
from deepparse.parser import FormattedParsedAddress


class TestFormattedComparedAddressesTags(TestCase):
    def test_givenIdenticalAddressesTags_whenCompareTags_thenReturnIdenticalComparisonReport(
        self,
    ):
        original_raw_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        original_parsed_address = [
            ("350", "StreetNumber"),
            ("rue des Lilas", "StreetName"),
            ("Ouest", "Orientation"),
            ("Quebec", "Municipality"),
            ("Quebec", "Province"),
            ("G1L 1B6", "PostalCode"),
        ]

        original_formatted_parsed_address = FormattedParsedAddress({original_raw_address: original_parsed_address})

        identical_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        identical_address_parsing_with_probs = [
            ("350", ("StreetNumber", 1.0)),
            ("rue", ("StreetName", 0.9987)),
            ("des", ("StreetName", 0.9993)),
            ("Lilas", ("StreetName", 0.8176)),
            ("Ouest", ("Orientation", 0.781)),
            ("Quebec", ("Municipality", 0.9768)),
            ("Quebec", ("Province", 1.0)),
            ("G1L", ("PostalCode", 0.9993)),
            ("1B6", ("PostalCode", 1.0)),
        ]

        identical_formatted_parsed_address = FormattedParsedAddress(
            {identical_address: identical_address_parsing_with_probs}
        )

        identical_formatted_compared_addresses_tags = FormattedComparedAddressesTags(
            first_address=original_formatted_parsed_address,
            second_address=identical_formatted_parsed_address,
            origin=("source", "deepparse using Bpemb"),
            with_prob=True,
        )

        expected_sentences = [
            "Comparison report of tags for parsed address: Identical\n\nRaw address: 350 rue des Lilas Ouest ",
            "Quebec Quebec G1L 1B6\n\n\nTags: \nsource:",
            "('StreetName', 0.8176))",
            "('1B6', ('PostalCode', 1.0))",
        ]

        actual = identical_formatted_compared_addresses_tags._comparison_report_builder()

        for expected_sentence in expected_sentences:
            self.assertIn(expected_sentence, actual)

    def test_givenNotEquivalentAddressesTags_whenCompareTags_thenReturnNotEquivalentComparisonReport(
        self,
    ):
        original_raw_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        original_raw_address_with_probs = [
            ("350", ("StreetNumber", 1.0)),
            ("rue", ("StreetName", 0.9987)),
            ("des", ("StreetName", 0.9993)),
            ("Lilas", ("StreetName", 0.8176)),
            ("Ouest", ("Orientation", 0.781)),
            ("Quebec", ("Municipality", 0.9768)),
            ("Quebec", ("Province", 1.0)),
            ("G1L", ("PostalCode", 0.9993)),
            ("1B6", ("PostalCode", 1.0)),
        ]

        original_formatted_parsed_address = FormattedParsedAddress(
            {original_raw_address: original_raw_address_with_probs}
        )

        # Not identical address with the preceding
        not_equivalent_address = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
        not_equivalent_address_parsing = [
            ("350", ("StreetNumber", 1.0)),
            ("rue", ("StreetName", 0.9987)),
            ("des", ("StreetName", 0.9993)),
            ("Lilas", ("StreetName", 0.8176)),
            ("Ouest", ("Municipality", 0.781)),
            ("Quebec", ("Municipality", 0.9768)),
            ("Quebec", ("Province", 1.0)),
            ("G1L", ("PostalCode", 0.9993)),
            ("1B6", ("PostalCode", 1.0)),
        ]

        not_equivalent_formatted_parsed_address = FormattedParsedAddress(
            {not_equivalent_address: not_equivalent_address_parsing}
        )

        not_equivalent_formatted_compared_addresses_raw_ = FormattedComparedAddressesTags(
            first_address=original_formatted_parsed_address,
            second_address=not_equivalent_formatted_parsed_address,
            origin=("source", "deepparse using Bpemb"),
            with_prob=True,
        )

        expected_sentences = [
            "Shared\nBlue: Belongs only to the source\nYellow: Belongs only to the deepparse using ",
            "\n\x1b[38;2;26;123;220mOuest\x1b[0m\n",
            "Municipality: \n\x1b[38;2;255;194;10mOuest \x1b[0m\x1b[38;2;255;255;255m",
            "Quebec\x1b[0m\n",
        ]

        actual = not_equivalent_formatted_compared_addresses_raw_._comparison_report_builder()
        for expected_sentence in expected_sentences:
            self.assertIn(expected_sentence, actual)


if __name__ == "__main__":
    unittest.main()
