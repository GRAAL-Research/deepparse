from unittest import TestCase

from deepparse.data_validation import is_whitespace_only, is_empty


class DataValidationTest(TestCase):
    def test_if_no_whitespace_address_when_is_white_space_return_false(self):
        an_address_not_whitespace_only = "an address"

        self.assertFalse(is_whitespace_only(an_address_not_whitespace_only))

        another_address_not_whitespace_only = "address"

        self.assertFalse(is_whitespace_only(another_address_not_whitespace_only))

    def test_if_whitespace_address_when_is_white_space_return_true(self):
        an_address_whitespace_only = " "

        self.assertTrue(is_whitespace_only(an_address_whitespace_only))

        another_address_whitespace_only = "  "

        self.assertTrue(is_whitespace_only(another_address_whitespace_only))

        a_last_address_whitespace_only = "       "

        self.assertTrue(is_whitespace_only(a_last_address_whitespace_only))

    def test_if_no_empty_address_when_is_empty_address_return_false(self):
        an_address_not_empty = "an address"

        self.assertFalse(is_empty(an_address_not_empty))

        another_address_not_empty = "address"

        self.assertFalse(is_empty(another_address_not_empty))

    def test_if_empty_address_when_is_empty_address_return_true(self):
        an_address_empty = ""

        self.assertTrue(is_empty(an_address_empty))

        another_address_empty = ''

        self.assertTrue(is_empty(another_address_empty))
