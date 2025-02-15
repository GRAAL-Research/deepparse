# pylint: disable=too-many-public-methods

from unittest import TestCase

from deepparse import validate_if_any_empty, validate_if_any_whitespace_only, validate_if_any_none
from deepparse.data_validation import (
    is_whitespace_only,
    is_empty,
    is_none,
    validate_if_any_multiple_consecutive_whitespace,
    is_multiple_consecutive_whitespace,
    is_newline,
    validate_if_any_newline_character,
)


class DataValidationTest(TestCase):
    def setUp(self):
        self.a_list_of_string_element = ["an address", "another address"]

    def test_integration_validate_if_any_empty_return_false(self):
        self.assertFalse(validate_if_any_empty(self.a_list_of_string_element))

    def test_integration_validate_if_any_whitespace_only_return_false(self):
        self.assertFalse(validate_if_any_whitespace_only(self.a_list_of_string_element))

    def test_integration_validate_if_any_none_return_false(self):
        self.assertFalse(validate_if_any_none(self.a_list_of_string_element))

    def test_integration_validate_if_any_multiple_consecutive_whitespace_return_false(self):
        self.assertFalse(validate_if_any_multiple_consecutive_whitespace(self.a_list_of_string_element))

    def test_integration_validate_if_any_newline_character_return_false(self):
        self.assertFalse(validate_if_any_newline_character(self.a_list_of_string_element))

    def test_integration_validate_if_any_empty_with_empty_return_true(self):
        a_list_of_string_element = ["an address", ""]
        self.assertTrue(validate_if_any_empty(a_list_of_string_element))

    def test_integration_validate_if_any_whitespace_only_with_whitespace_return_true(self):
        a_list_of_string_element = ["an address", "  "]
        self.assertTrue(validate_if_any_whitespace_only(a_list_of_string_element))

    def test_integration_validate_if_any_none_with_none_return_true(self):
        a_list_of_string_element = ["an address", None]
        self.assertTrue(validate_if_any_none(a_list_of_string_element))

    def test_integration_validate_if_any_multiple_consecutive_whitespace_with_multiple_whitespace_return_true(self):
        a_list_of_string_element = ["an address", "an  address"]
        self.assertTrue(validate_if_any_multiple_consecutive_whitespace(a_list_of_string_element))

        a_list_of_string_element = ["an address", "an   address"]
        self.assertTrue(validate_if_any_multiple_consecutive_whitespace(a_list_of_string_element))

        a_list_of_string_element = ["an address", "an  address", "an   address"]
        self.assertTrue(validate_if_any_multiple_consecutive_whitespace(a_list_of_string_element))

    def test_integration_validate_if_any_newline_character_with_newline_return_true(self):
        a_list_of_string_element = ["an address", "an address\n"]
        self.assertTrue(validate_if_any_newline_character(a_list_of_string_element))

        a_list_of_string_element = ["an address", "an\n address"]
        self.assertTrue(validate_if_any_newline_character(a_list_of_string_element))

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

    def test_if_none_when_is_none_return_true(self):
        an_address_not_empty = None

        self.assertTrue(is_none(an_address_not_empty))

        another_address_not_empty = None

        self.assertTrue(is_none(another_address_not_empty))

    def test_if_no_consecutive_whitespace_when_is_multiple_consecutive_whitespace_return_false(self):
        an_address_not_empty = "an address"

        self.assertFalse(is_multiple_consecutive_whitespace(an_address_not_empty))

        another_address_not_empty = "address"

        self.assertFalse(is_multiple_consecutive_whitespace(another_address_not_empty))

    def test_if_consecutive_whitespace_when_is_multiple_consecutive_whitespace_return_true(self):
        an_address_not_empty = "an  address"

        self.assertTrue(is_multiple_consecutive_whitespace(an_address_not_empty))

        another_address_not_empty = "address  "

        self.assertTrue(is_multiple_consecutive_whitespace(another_address_not_empty))

    def test_if_no_newline_when_is_newline_return_false(self):
        an_address_not_empty = "an address"

        self.assertFalse(is_newline(an_address_not_empty))

        another_address_not_empty = "address"

        self.assertFalse(is_newline(another_address_not_empty))

    def test_if_newline_when_is_newline_return_true(self):
        an_address_not_empty = "an address\n"

        self.assertTrue(is_newline(an_address_not_empty))

        another_address_not_empty = "address \n"

        self.assertTrue(is_newline(another_address_not_empty))
