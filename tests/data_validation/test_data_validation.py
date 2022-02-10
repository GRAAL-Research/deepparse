from unittest import TestCase

from deepparse import validate_if_any_empty, validate_if_any_whitespace_only, validate_if_any_none
from deepparse.data_validation import is_whitespace_only, is_empty, is_none


class DataValidationTest(TestCase):
    def test_integration_validate_if_any_empty(self):
        a_list_of_string_element = ["an address", "another address"]
        self.assertFalse(validate_if_any_empty(a_list_of_string_element))

    def test_integration_validate_if_any_whitespace_only(self):
        a_list_of_string_element = ["an address", "another address"]
        self.assertFalse(validate_if_any_whitespace_only(a_list_of_string_element))

    def test_integration_validate_if_any_none(self):
        a_list_of_string_element = ["an address", "another address"]
        self.assertFalse(validate_if_any_none(a_list_of_string_element))

    def test_integration_validate_if_any_empty_with_empty_return_true(self):
        a_list_of_string_element = ["an address", ""]
        self.assertTrue(validate_if_any_empty(a_list_of_string_element))

    def test_integration_validate_if_any_whitespace_only_with_whitespace_return_true(self):
        a_list_of_string_element = ["an address", "  "]
        self.assertTrue(validate_if_any_whitespace_only(a_list_of_string_element))

    def test_integration_validate_if_any_none_with_none_return_true(self):
        a_list_of_string_element = ["an address", None]
        self.assertTrue(validate_if_any_none(a_list_of_string_element))

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

    def test_if_no_none_when_is_none_return_false(self):
        an_address_not_empty = "an address"

        self.assertFalse(is_none(an_address_not_empty))

        another_address_not_empty = "address"

        self.assertFalse(is_none(another_address_not_empty))

    def test_if_no_none_when_is_none_return_true(self):
        an_address_not_empty = None

        self.assertTrue(is_none(an_address_not_empty))

        another_address_not_empty = None

        self.assertTrue(is_none(another_address_not_empty))
