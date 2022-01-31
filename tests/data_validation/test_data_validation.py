from unittest import TestCase

from deepparse.data_validation.data_validation import is_whitespace_only_address


class DataValidationTest(TestCase):
    def test_if_no_whitespace_address_when_is_white_space_return_false(self):
        an_address_not_whitespace_only = "an address"

        self.assertFalse(is_whitespace_only_address(an_address_not_whitespace_only))

        another_address_not_whitespace_only = "address"

        self.assertFalse(is_whitespace_only_address(another_address_not_whitespace_only))

    def test_if_whitespace_address_when_is_white_space_return_true(self):
        an_address_whitespace_only = " "

        self.assertTrue(is_whitespace_only_address(an_address_whitespace_only))

        another_address_whitespace_only = "  "

        self.assertTrue(is_whitespace_only_address(another_address_whitespace_only))

        a_last_address_whitespace_only = "       "

        self.assertTrue(is_whitespace_only_address(a_last_address_whitespace_only))
