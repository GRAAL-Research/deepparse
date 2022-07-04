from unittest import TestCase

from deepparse.preprocessing import AddressCleaner


class AddressCleanerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_clean_address = "350 rue des lilas ouest québec québec g1l 1b6"
        cls.a_dirty_address_with_commas = "350 rue des lilas , ouest ,québec québec, g1l 1b6"
        cls.a_commas_separated_address = "350, rue des lilas, ouest, québec, québec, g1l 1b6"
        cls.a_dirty_address_with_uppercase = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        cls.a_dirty_address_with_whitespaces = "350     rue des Lilas Ouest Québec Québec G1L 1B6"

        cls.an_address_with_hyphen_split_address_components = "3-350 rue des lilas ouest"
        cls.a_unit_clean_address = "3 350 rue des lilas ouest"

        cls.an_address_with_hyphen_split_address_components_with_hyphen_city = "3-350 rue des lilas ouest, saint-jean"
        cls.a_unit_hyphen_city_name_clean_address = "3 350 rue des lilas ouest saint-jean"

        cls.a_unit_with_letter_hyphen_split = "3a-350 rue des lilas ouest saint-jean"
        cls.a_unit_with_letter_hyphen_split_clean_address = "3a 350 rue des lilas ouest saint-jean"

        cls.a_unit_with_letter_only_hyphen_split = "a-350 rue des lilas ouest saint-jean"
        cls.a_unit_with_letter_only_hyphen_split_clean_address = "a 350 rue des lilas ouest saint-jean"

        cls.a_street_number_with_letter_hyphen_split = "3-350a rue des lilas ouest saint-jean"
        cls.a_street_number_with_letter_hyphen_split_clean_address = "3 350a rue des lilas ouest saint-jean"

        cls.letters_hyphen_address = "3a-350b rue des lilas ouest saint-jean"
        cls.letters_hyphen_address_split_clean_address = "3a 350b rue des lilas ouest saint-jean"

        cls.address_cleaner = AddressCleaner()

    def test_givenACleanAddress_whenCleaningAddress_thenShouldNotMakeAnyChange(self):
        cleaned_address = self.address_cleaner.clean([self.a_clean_address])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenADirtyAddressWithCommas_whenCleaningAddress_thenShouldRemoveCommas(
        self,
    ):
        cleaned_address = self.address_cleaner.clean([self.a_dirty_address_with_commas])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

        cleaned_address = self.address_cleaner.clean([self.a_commas_separated_address])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenADirtyAddressWithUppercase_whenCleaningAddress_thenShouldLower(self):
        cleaned_address = self.address_cleaner.clean([self.a_dirty_address_with_uppercase])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenADirtyAddressWithWhitespaces_whenCleaningAddress_thenShouldRemoveWhitespaces(
        self,
    ):
        cleaned_address = self.address_cleaner.clean([self.a_dirty_address_with_whitespaces])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenMultipleDirtyAddresses_whenCleaningAddresses_thenShouldCleanAllAddresses(
        self,
    ):
        cleaned_address = self.address_cleaner.clean(
            [self.a_dirty_address_with_whitespaces, self.a_dirty_address_with_uppercase]
        )

        self.assertEqual(self.a_clean_address, cleaned_address[0])
        self.assertEqual(self.a_clean_address, cleaned_address[1])

    def test_givenAHyphenUnitStreetNumberAddress_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        self.address_cleaner = AddressCleaner(with_hyphen_split=True)

        cleaned_address = self.address_cleaner.clean([self.an_address_with_hyphen_split_address_components])

        self.assertEqual(self.a_unit_clean_address, cleaned_address[0])

    def test_givenAHyphenUnitAndCityAddress_whenCleaningAddress_thenShouldReplaceUnitStreetNumberHyphenWithWhiteSpace(
        self,
    ):
        self.address_cleaner = AddressCleaner(with_hyphen_split=True)

        cleaned_address = self.address_cleaner.clean(
            [self.an_address_with_hyphen_split_address_components_with_hyphen_city]
        )

        self.assertEqual(self.a_unit_hyphen_city_name_clean_address, cleaned_address[0])

    def test_givenAnAlphabeticalUnitStreetNumberHyphen_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        self.address_cleaner = AddressCleaner(with_hyphen_split=True)

        cleaned_address = self.address_cleaner.clean([self.a_unit_with_letter_hyphen_split])

        self.assertEqual(self.a_unit_with_letter_hyphen_split_clean_address, cleaned_address[0])

    def test_givenAnAlphabeticalOnlyUnitHyphen_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        self.address_cleaner = AddressCleaner(with_hyphen_split=True)

        cleaned_address = self.address_cleaner.clean([self.a_unit_with_letter_only_hyphen_split])

        self.assertEqual(self.a_unit_with_letter_only_hyphen_split_clean_address, cleaned_address[0])

    def test_givenAnAlphabeticalStreetNumberUnitHyphen_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(self):
        self.address_cleaner = AddressCleaner(with_hyphen_split=True)

        cleaned_address = self.address_cleaner.clean([self.a_street_number_with_letter_hyphen_split])

        self.assertEqual(self.a_street_number_with_letter_hyphen_split_clean_address, cleaned_address[0])

    def test_givenAnAlphabeticalComponentsStreetNumberUnit_whenCleaningAddress_thenShouldReplaceHyphenWithWhiteSpace(
        self,
    ):
        self.address_cleaner = AddressCleaner(with_hyphen_split=True)

        cleaned_address = self.address_cleaner.clean([self.letters_hyphen_address])

        self.assertEqual(self.letters_hyphen_address_split_clean_address, cleaned_address[0])
