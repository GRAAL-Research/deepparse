from unittest import TestCase

from deepparse.preprocessing import AddressCleaner


class AddressCleanerTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_clean_address = "350 rue des lilas ouest québec québec g1l 1b6"
        cls.a_dirty_address_with_commas = "350 rue des lilas , ouest ,québec québec, g1l 1b6"
        cls.a_dirty_address_with_uppercase = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
        cls.a_dirty_address_with_whitespaces = "350     rue des Lilas Ouest Québec Québec G1L 1B6"

    def test_givenACleanAddress_whenCleaningAddress_thenShouldNotMakeAnyChange(self):
        cleaned_address = AddressCleaner().clean([self.a_clean_address])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenADirtyAddressWithCommas_whenCleaningAddress_thenShouldRemoveCommas(self):
        cleaned_address = AddressCleaner().clean([self.a_dirty_address_with_commas])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenADirtyAddressWithUppercase_whenCleaningAddress_thenShouldLower(self):
        cleaned_address = AddressCleaner().clean([self.a_dirty_address_with_uppercase])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenADirtyAddressWithWhitespaces_whenCleaningAddress_thenShouldRemoveWhitespaces(self):
        cleaned_address = AddressCleaner().clean([self.a_dirty_address_with_whitespaces])

        self.assertEqual(self.a_clean_address, cleaned_address[0])

    def test_givenMultipleDirtyAddresses_whenCleaningAddresses_thenShouldCleanAllAddresses(self):
        cleaned_address = AddressCleaner().clean(
            [self.a_dirty_address_with_whitespaces, self.a_dirty_address_with_uppercase])

        self.assertEqual(self.a_clean_address, cleaned_address[0])
        self.assertEqual(self.a_clean_address, cleaned_address[1])
