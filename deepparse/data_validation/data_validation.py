import re
from typing import List

# Regular expression to find if a string contains consecutive whitespace
consecutive_whitespace_regular_expression = re.compile(r"\s{2,}")

# Regular expression to find if a string contains a newline character
newline_regular_expression = re.compile(r"\n")


def validate_if_any_empty(string_elements: List) -> bool:
    """
    Return ``True`` if one of the string elements is empty. For example, the second element in the following list is
    an empty address: ``["An address", "", "Another address"]``. Thus, it will return ``True``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_empty(string_element) for string_element in string_elements)


def validate_if_any_whitespace_only(string_elements: List) -> bool:
    """
    Return ``True`` if one of the string elements is only whitespace. For example, the second element in the
    following list is only whitespace: ``["An address", " ", "Another address"]``. Thus, it will return ``True``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_whitespace_only(string_element) for string_element in string_elements)


def validate_if_any_none(string_elements: List) -> bool:
    """
    Return ``True`` if one string element is a ``None`` value. For example, the second element in the following
    list is a ``None`` value: ``["An address", None, "Another address"]``. Thus, it will return ``True``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_none(string_element) for string_element in string_elements)


def validate_if_any_multiple_consecutive_whitespace(string_elements: List) -> bool:
    """
    Return ``True`` if one string element include multiple consecutive_whitespace.
    For example, the second element in the following list has two consecutive whitespace:
    ``["An address", "An  address", "Another address"]``. Thus, it will return ``True``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_multiple_consecutive_whitespace(string_element) for string_element in string_elements)


def validate_if_any_newline_character(string_elements: List) -> bool:
    """
    Return ``True`` if one string element include a newline character.
    For example, the second element in the following list include a newline character.
    ``["An address", "An address\n", "Another address"]``. Thus, it will return ``True``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_newline(string_element) for string_element in string_elements)


def is_whitespace_only(string_element: str) -> bool:
    """
    Validate if a string is composed of only whitespace.

    Args:
        string_element (str): A string to validate.

    Return:
        Either or not, the string is composed only of whitespace or not.
    """
    return len(string_element.strip(" ").split()) == 0


def is_empty(string_element: str) -> bool:
    """
    Validate if a string is empty.

    Args:
        string_element (str): A string to validate.

    Return:
        Either or not, the string is empty.
    """
    return len(string_element) == 0


def is_none(string_element: str) -> bool:
    """
    Validate if a string is a None.

    Args:
        string_element (str): A string to validate.

    Return:
        Either or not, the string is a None type.
    """
    return string_element is None


def is_multiple_consecutive_whitespace(string_element: str) -> bool:
    """
    Validate if a string include consecutive whitespace. Consecutive whitespace will break matching between the
    address components and the tags during splitting.

    Args:
        string_element (str): A string to validate.

    Return:
        Either or not, the string include consecutive whitespace.
    """
    return consecutive_whitespace_regular_expression.search(string_element) is not None


def is_newline(string_element: str) -> bool:
    """
    Validate if a string include a newline character.

    Args:
        string_element (str): A string to validate.

    Return:
        Either or not, the string include a newline character.
    """
    return newline_regular_expression.search(string_element) is not None
