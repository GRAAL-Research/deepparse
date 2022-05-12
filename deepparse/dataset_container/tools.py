from typing import List

from deepparse.data_validation import validate_if_any_empty, validate_if_any_whitespace_only


def former_python_list(tags: str) -> List:
    """
    Function to parse a former Python list exported in a string format.

    Args:
        tags (str): A tag set string to parse.

    Return:
        A list of the parsed tag set.
    """
    # We remove the [ and ] of the list.
    # Then, we split each element using a comma as separator.
    # Finally, since some case the element are separated by a comma (e.g. element1,element2)
    # or a comma and a whitespace (e.g. element1, element2), we strip the whitespace on all tags to
    # remove the trailing whitespace when element are separated by a coma and a whitespace.
    # To fix https://github.com/GRAAL-Research/deepparse/issues/124.
    return [tag.strip() for tag in tags.replace("[", "").replace("]", "").replace("'", "").split(",")]


def validate_column_names(column_names: List[str]) -> bool:
    """
    Function validate if element of a list of column name are valid.

    Args:
        column_names (List[str]): A list of column names.

    Return:
        Either or not, the colum name are valid.
    """
    improper_column_names = False
    if validate_if_any_empty(column_names) or validate_if_any_whitespace_only(column_names):
        improper_column_names = True
    return improper_column_names
