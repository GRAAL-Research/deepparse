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
