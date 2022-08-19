import os
import shutil
import warnings
from pathlib import Path
from typing import List

import poutyne
import requests
from requests import HTTPError
from urllib3.exceptions import MaxRetryError

from .data_error import DataError
from .data_validation import (
    validate_if_any_none,
    validate_if_any_whitespace_only,
    validate_if_any_empty,
)

BASE_URL = "https://graal.ift.ulaval.ca/public/deepparse/{}.{}"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

# Status code starting in the 4xx are client error status code.
# That is Deepparse, server problem (e.g. Deepparse server is offline).
HTTP_CLIENT_ERROR_STATUS_CODE = 400
# Status code starting in the 5xx are the next range status code.
NEXT_RANGE_STATUS_CODE = 500


def latest_version(model: str, cache_path: str, verbose: bool) -> bool:
    """
    Verify if the local model is the latest.
    """
    # Reading of the actual local version
    with open(os.path.join(cache_path, model + ".version"), encoding="utf-8") as local_model_hash_file:
        local_model_hash_version = local_model_hash_file.readline()
    try:
        # We create a temporary directory for the server-side version file
        tmp_cache = os.path.join(cache_path, "tmp")
        os.makedirs(tmp_cache, exist_ok=True)

        download_from_public_repository(model, tmp_cache, "version")

        # Reading of the server-side version
        with open(os.path.join(tmp_cache, model + ".version"), encoding="utf-8") as remote_model_hash_file:
            remote_model_hash_version = remote_model_hash_file.readline()

        is_latest_version = local_model_hash_version.strip() == remote_model_hash_version.strip()

    except HTTPError as exception:  # HTTP connection error handling
        if HTTP_CLIENT_ERROR_STATUS_CODE <= exception.response.status_code < NEXT_RANGE_STATUS_CODE:
            # Case where Deepparse server is down.
            if verbose:
                warnings.warn(
                    f"We where not able to verify the cached model in the cache directory {cache_path}. It seems like"
                    f"Deepparse server is not available at the moment. We recommend to attempt to verify "
                    f"the model version another time using our download CLI function."
                )
            # The is_lastest_version is set to True even if we were not able to validate the version. We do so not to
            # block the rest of the process.
            is_latest_version = True
        else:
            # We re-raise the exception if the status_code is not in the two ranges we are interested in
            # (local server or remote server error).
            raise
    except MaxRetryError:
        # Case where the user does not have an Internet connection. For example, one can run it in a
        # Docker container not connected to the Internet.
        if verbose:
            warnings.warn(
                f"We where not able to verify the cached model in the cache directory {cache_path}. It seems like"
                f"you are not connected to the Internet. We recommend to verify if you have the latest using our "
                f"download CLI function."
            )
        # The is_lastest_version is set to True even if we were not able to validate the version. We do so not to
        # block the rest of the process.
        is_latest_version = True
    finally:
        # Cleaning the temporary directory
        if os.path.exists(tmp_cache):
            shutil.rmtree(tmp_cache)

    return is_latest_version


def download_from_public_repository(file_name: str, saving_dir: str, file_extension: str) -> None:
    """
    Simple function to download the content of a file from Deepparse public repository.
    The repository URL string is  `'https://graal.ift.ulaval.ca/public/deepparse/{}.{}'``
    where the first bracket is the file name and the second is the file extension.
    """
    url = BASE_URL.format(file_name, file_extension)
    r = requests.get(url, timeout=5)
    r.raise_for_status()  # Raise exception
    Path(saving_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(saving_dir, f"{file_name}.{file_extension}"), "wb") as file:
        file.write(r.content)


def download_weights(model: str, saving_dir: str, verbose: bool = True) -> None:
    """
    Function to download the pretrained weights of the models.
    Args:
        model: The network type (i.e. fasttext or bpemb).
        saving_dir: The path to the saving directory.
        verbose (bool): Turn on/off the verbosity of the model. The default value is True.
    """
    if verbose:
        print(f"Downloading the weights for the network {model}.")
    download_from_public_repository(model, saving_dir, "ckpt")
    download_from_public_repository(model, saving_dir, "version")


def handle_poutyne_version() -> str:
    """
    Handle the retrieval of the major and minor part of the Poutyne version
    """
    full_version = poutyne.version.__version__
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
    version_components = handle_poutyne_version().split(".")

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
