from typing import List

from deepparse import is_whitespace_only
from deepparse.data_validation import is_empty


def former_python_list(tags: str) -> List:
    """
    Function to parse a former Python list exported in a string format.

    Args:
        tags (str): A tag set string to parse.

    Return:
        A list of the parsed tag set.
    """
    return tags.replace("[", "").replace("]", "").replace("'", "").split(", ")


def comma_separated_list_reformat(tags: str) -> List:
    """
    Function to parse a comma separated "list" of tag.

    Args:
        tags (str): A tag set string to parse.

    Return:
        A list of the parsed tag set.
    """
    return tags.split(", ")


def validate_column_names(column_names: List[str]) -> bool:
    """
    Function validate if element of a list of column name are valid.

    Args:
        column_names (List[str]): A list of column names.

    Return:
        Either or not, the colum name are valid.
    """
    improper_column_names = False
    if is_empty_any_column_name(column_names) or is_whitespace_only_any(column_names):
        improper_column_names = True
    return improper_column_names


def is_empty_any_column_name(column_names: List[str]) -> bool:
    """
    Function validate if any element of a list of column name are empty.

    Args:
        column_names (List[str]): A list of column names.

    Return:
        Either or not, an element of the list is an empty string.
    """

    return any((is_empty(column_name) for column_name in column_names))


def is_whitespace_only_any(column_names: List[str]) -> bool:
    """
    Function validate if any element of a list of column name are whitespace only.

    Args:
        column_names (List[str]): A list of column names.

    Return:
        Either or not, an element of the list is a whitespace only string.
    """
    return any((is_whitespace_only(column_name) for column_name in column_names))
