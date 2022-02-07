from typing import List


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
