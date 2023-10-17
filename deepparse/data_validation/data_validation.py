from typing import List


def validate_if_any_empty(string_elements: List) -> bool:
    """
    Return ``True`` if one of the string elements is empty. For example, the second element in the following list is
    an empty address: ``["An address", "", "Another address"]``. Thus, it will return ``False``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_empty(string_element) for string_element in string_elements)


def validate_if_any_whitespace_only(string_elements: List) -> bool:
    """
    Return ``True`` if one of the string elements is only whitespace. For example, the second element in the
    following list is only whitespace: ``["An address", " ", "Another address"]``. Thus, it will return ``False``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_whitespace_only(string_element) for string_element in string_elements)


def validate_if_any_none(string_elements: List) -> bool:
    """
    Return ``True`` if one string element is a ``None`` value. For example, the second element in the following
    list is a ``None`` value: ``["An address", None, "Another address"]``. Thus, it will return ``False``.

    Args:
        string_elements (list): A list of strings to validate.
    """
    return any(is_none(string_element) for string_element in string_elements)


def is_whitespace_only(a_string: str) -> bool:
    """
    Validate if a string is composed of only whitespace.

    Args:
        a_string (str): A string to validate.

    Return:
        Either or not, the string is composed only of whitespace or not.
    """
    return len(a_string.strip(" ").split()) == 0


def is_empty(a_string: str) -> bool:
    """
    Validate if a string is empty.

    Args:
        a_string (str): A string to validate.

    Return:
        Either or not, the string is empty.
    """
    return len(a_string) == 0


def is_none(a_string: str) -> bool:
    """
    Validate if a string is a None.

    Args:
        a_string (str): A string to validate.

    Return:
        Either or not, the string is a None type.
    """
    return a_string is None
