from typing import List

import poutyne

from .data_validation import (
    validate_if_any_none,
    validate_if_any_whitespace_only,
    validate_if_any_empty,
)
from .errors.data_error import DataError


def extract_package_version(package) -> str:
    """
    Handle the retrieval of the major and minor version part of a Python package.
    """
    full_version = package.version.__version__
    components_parts = full_version.split(".")
    major = components_parts[0]
    minor = components_parts[1]
    version = f"{major}.{minor}"
    return version


def valid_poutyne_version(min_major: int = 1, min_minor: int = 2) -> bool:
    """
    Validate Poutyne version is greater than min_major.min_minor for using a str checkpoint. Some version before
    does not support all the features we need. By default, min_major.min_minor equal version 1.2 which is the
    lowest version we can use.
    """
    version_components = extract_package_version(package=poutyne).split(".")

    major = int(version_components[0])
    minor = int(version_components[1])

    if major > min_major:
        is_valid_poutyne_version = True
    else:
        is_valid_poutyne_version = major >= min_major and minor >= min_minor

    return is_valid_poutyne_version


def validate_data_to_parse(addresses_to_parse: List) -> None:
    """
    Validation tests on the addresses to parse to respect the following two criteria:
        - addresses are not tuple,
        - no addresses are None value,
        - no addresses are empty strings, and
        - no addresses are whitespace-only strings.
    """
    if isinstance(addresses_to_parse[0], tuple):
        raise DataError(
            "Addresses to parsed are tuples. They need to be a list of string. Are you using training data?"
        )
    if validate_if_any_none(addresses_to_parse):
        raise DataError("Some addresses are None value.")
    if validate_if_any_empty(addresses_to_parse):
        raise DataError("Some addresses are empty.")
    if validate_if_any_whitespace_only(addresses_to_parse):
        raise DataError("Some addresses only include whitespace thus cannot be parsed.")
