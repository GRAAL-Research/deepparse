from unittest import TestCase

from deepparse.pre_processing import (
    coma_cleaning,
    lower_cleaning,
    trailing_whitespace_cleaning,
    hyphen_cleaning,
    double_whitespaces_cleaning,
)


class AddressClearnerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_clean_address = "350 rue des lilas ouest québec québec g1l 1b6"
        cls.a_dirty_address_with_commas = "350 rue des lilas , ouest ,québec québec, g1l 1b6"
        cls.a_commas_separated_address = "350, rue des lilas, ouest, québec, québec, g1l 1b6"
        cls.a_dirty_address_with_uppercase = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        cls.a_dirty_address_with_trailing_whitespaces = "350 rue des lilas ouest québec québec g1l 1b6 "
        cls.a_dirty_address_with_whitespaces = "350     rue des lilas ouest québec québec g1l 1b6"

        cls.an_address_with_hyphen_split_address_components = "3-350 rue des lilas ouest"
        cls.a_unit_clean_address = "3 350 rue des lilas ouest"

        cls.an_address_with_hyphen_split_address_components_with_hyphen_city = "3-350 rue des lilas ouest saint-jean"
        cls.a_unit_hyphen_city_name_clean_address = "3 350 rue des lilas ouest saint-jean"

        cls.a_unit_with_letter_hyphen_split = "3a-350 rue des lilas ouest saint-jean"
        cls.a_unit_with_letter_hyphen_split_clean_address = "3a 350 rue des lilas ouest saint-jean"

        cls.a_unit_with_letter_only_hyphen_split = "a-350 rue des lilas ouest saint-jean"
        cls.a_unit_with_letter_only_hyphen_split_clean_address = "a 350 rue des lilas ouest saint-jean"

        cls.a_street_number_with_letter_hyphen_split = "3-350a rue des lilas ouest saint-jean"
        cls.a_street_number_with_letter_hyphen_split_clean_address = "3 350a rue des lilas ouest saint-jean"

        cls.letters_hyphen_address = "3a-350b rue des lilas ouest saint-jean"
        cls.letters_hyphen_address_split_clean_address = "3a 350b rue des lilas ouest saint-jean"

    def test_givenADirtyAddressWithCommas_whenComaCleaning_thenShouldRemoveCommas(
        self,
    ):
        cleaned_address = coma_cleaning(self.a_commas_separated_address)

        self.assertEqual(self.a_clean_address, cleaned_address)

    def test_givenADirtyAddressWithUppercase_whenLowerCleaning_thenShouldLower(self):
        cleaned_address = lower_cleaning(self.a_dirty_address_with_uppercase)

        self.assertEqual(self.a_clean_address, cleaned_address)

    def test_givenADirtyAddressWithWhitespaces_whenTrailingWhitespaceCleaning_thenShouldRemoveWhitespaces(
        self,
    ):
        cleaned_address = trailing_whitespace_cleaning(self.a_dirty_address_with_trailing_whitespaces)

        self.assertEqual(self.a_clean_address, cleaned_address)

    def test_givenADirtyAddressWithWhitespacesInAddress_whenDoubleWhitespacesCleaning_thenShouldRemoveWhitespaces(
        self,
    ):
        cleaned_address = double_whitespaces_cleaning(self.a_dirty_address_with_whitespaces)

        self.assertEqual(self.a_clean_address, cleaned_address)

    def test_givenAHyphenUnitStreetNumberAddress_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        cleaned_address = hyphen_cleaning(self.an_address_with_hyphen_split_address_components)

        self.assertEqual(self.a_unit_clean_address, cleaned_address)

    def test_givenAHyphenUnitAndCityAddress_whenCleaningAddress_thenShouldReplaceUnitStreetNumberHyphenWithWhiteSpace(
        self,
    ):
        cleaned_address = hyphen_cleaning(self.an_address_with_hyphen_split_address_components_with_hyphen_city)

        self.assertEqual(self.a_unit_hyphen_city_name_clean_address, cleaned_address)

    def test_givenAnAlphabeticalUnitStreetNumberHyphen_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        cleaned_address = hyphen_cleaning(self.a_unit_with_letter_hyphen_split)

        self.assertEqual(self.a_unit_with_letter_hyphen_split_clean_address, cleaned_address)

    def test_givenAnAlphabeticalOnlyUnitHyphen_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        cleaned_address = hyphen_cleaning(self.a_unit_with_letter_only_hyphen_split)

        self.assertEqual(self.a_unit_with_letter_only_hyphen_split_clean_address, cleaned_address)

    def test_givenAnAlphabeticalStreetNumberUnitHyphen_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        cleaned_address = hyphen_cleaning(self.a_street_number_with_letter_hyphen_split)

        self.assertEqual(self.a_street_number_with_letter_hyphen_split_clean_address, cleaned_address)

    def test_givenAnAlphabeticalComponentsStreetNumberUnit_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(
        self,
    ):
        cleaned_address = hyphen_cleaning(self.letters_hyphen_address)

        self.assertEqual(self.letters_hyphen_address_split_clean_address, cleaned_address)
