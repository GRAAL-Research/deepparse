import re


def coma_cleaning(address: str) -> str:
    """
    Pre-processor to remove coma. It is based on `issue 56 <https://github.com/GRAAL-Research/deepparse/issues/56>`_.

    Args:
        address: The address to apply coma cleaning on.

    Return:
        The coma-cleaned address.
    """
    return address.replace(",", "")


def lower_cleaning(address: str) -> str:
    """
    Pre-processor to lowercase an address since the original training data was in lowercase.

    Args:
        address: The address to apply coma cleaning on.

    Return:
        The lowercase address.
    """
    return address.lower()


# The first group is the unit, and the second is the street number.
# Both include letters since they can include letters in some countries. For example,
# unit 3a or address 305a.
hyphen_splitted_unit_and_street_number_regex = r"^([0-9]*[a-z]?)-([0-9]*[a-z]?) "


def hyphen_cleaning(address: str) -> str:
    """
    Pre-processor to clean hyphen between the street number and unit in an address. Since some addresses use the
    hyphen to split the unit and street address, we replace the hyphen with whitespaces to allow a
    proper splitting of the address. For example, the proper parsing of the address 3-305 street name is
    Unit: 3, StreetNumber: 305, StreetName: street name.

    See `issue 137 <https://github.com/GRAAL-Research/deepparse/issues/137>`_ for more details.

    The regular expression use to clean hyphen is the following ``"^([0-9]*[a-z]?)-([0-9]*[a-z]?) "``.
    The first group is the unit, and the second is the street number. Both include letters since they can include
    letters in some countries. For example, unit 3a or address 305a.

    Note: the hyphen is also used in some cities' names, such as Saint-Jean; thus, we use regex to detect
    the proper hyphen to replace.

    Args:
        address: The address to apply coma cleaning on.

    Return:
        The lowercase address.
    """
    return re.sub(hyphen_splitted_unit_and_street_number_regex, r"\1 \2 ", address)
