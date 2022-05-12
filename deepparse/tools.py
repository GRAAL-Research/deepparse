import os
import warnings
from typing import List

import poutyne
import requests

from .data_error import DataError
from .data_validation import (
    validate_if_any_none,
    validate_if_any_whitespace_only,
    validate_if_any_empty,
)

BASE_URL = "https://graal.ift.ulaval.ca/public/deepparse/{}.{}"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")


def latest_version(model: str, cache_path: str) -> bool:
    """
    Verify if the local model is the latest.
    """
    with open(os.path.join(cache_path, model + ".version"), encoding="utf-8") as local_model_hash_file:
        local_model_hash_version = local_model_hash_file.readline()
    download_from_url(model, cache_path, "version")
    with open(os.path.join(cache_path, model + ".version"), encoding="utf-8") as remote_model_hash_file:
        remote_model_hash_version = remote_model_hash_file.readline()
    return local_model_hash_version.strip() == remote_model_hash_version.strip()


def download_from_url(file_name: str, saving_dir: str, file_extension: str):
    """
    Simple function to download the content of a file from a distant repository.
    The repository URL string is  ̀`'https://graal.ift.ulaval.ca/public/deepparse/{}.{}'``
    where the first bracket is the file name and the second is the file extension.
    """
    url = BASE_URL.format(file_name, file_extension)
    r = requests.get(url, timeout=5)
    r.raise_for_status()  # Raise exception
    os.makedirs(saving_dir, exist_ok=True)
    with open(os.path.join(saving_dir, f"{file_name}.{file_extension}"), "wb") as file:
        file.write(r.content)


def download_weights(model: str, saving_dir: str, verbose: bool = True) -> None:
    """
    Function to download the pre-trained weights of the models.
    Args:
        model: The network type (i.e. fasttext or bpemb).
        saving_dir: The path to the saving directory.
        verbose (bool): Turn on/off the verbosity of the model. The default value is True.
    """
    if verbose:
        print(f"Downloading the weights for the network {model}.")
    download_from_url(model, saving_dir, "ckpt")
    download_from_url(model, saving_dir, "version")


def handle_poutyne_version() -> float:
    """
    Handle the retrieval of the major and minor part of the Poutyne version
    """
    full_version = poutyne.version.__version__
    components_parts = full_version.split(".")
    major = components_parts[0]
    minor = components_parts[1]
    version = f"{major}.{minor}"
    return version


def valid_poutyne_version():
    """
    Validate Poutyne version is greater than 1.2 for using a str checkpoint. Version before does not support that
    feature.
    """
    version_components = handle_poutyne_version().split(".")

    major = int(version_components[0])
    minor = int(version_components[1])

    return major >= 1 and minor >= 2


def handle_pre_trained_checkpoint(model_type_checkpoint: str) -> str:
    """
    Handle the checkpoint formatting for pre trained models.
    """
    if not valid_poutyne_version():
        raise NotImplementedError(
            f"To load the pre-trained {model_type_checkpoint} model, you need to have a Poutyne version"
            "greater than 1.1 (>1.1)"
        )
    model_path = os.path.join(CACHE_PATH, f"{model_type_checkpoint}.ckpt")

    if not os.path.isfile(model_path):
        download_weights(model_type_checkpoint, CACHE_PATH, verbose=True)
    elif not latest_version(model_type_checkpoint, cache_path=CACHE_PATH):
        warnings.warn(
            "A newer model of fasttext is available, you can download it using the download script.",
            UserWarning,
        )
    checkpoint = os.path.join(CACHE_PATH, f"{model_type_checkpoint}.ckpt")
    return checkpoint


def handle_model_path(checkpoint: str) -> str:
    """
    Handle the validity of path.
    """
    if checkpoint in ("fasttext", "bpemb"):
        checkpoint = handle_pre_trained_checkpoint(checkpoint)
    elif isinstance(checkpoint, str) and checkpoint.endswith(".ckpt"):
        if not valid_poutyne_version():
            raise NotImplementedError(
                "To load a string path to a model, you need to have a Poutyne version" "greater than 1.1 (>1.1)"
            )
    else:
        raise ValueError(
            "The checkpoint is not valid. Can be a path in a string format (e.g. 'a_path_.ckpt'), "
            "'fasttext' or 'bpemb'."
        )

    return checkpoint


def validate_data_to_parse(addresses_to_parse: List) -> None:
    """
    Validation tests on the addresses to parse to respect the following two criteria:
        - addresses are not tuple,
        - no addresses are None value,
        - no addresses are empty strings, and
        - no addresses are whitespace-only strings.
    """
    if isinstance(addresses_to_parse[0], tuple):
        DataError("Addresses to parsed are tuples. They need to be a list of string. Are you using training data?")
    if validate_if_any_none(addresses_to_parse):
        raise DataError("Some addresses are None value.")
    if validate_if_any_empty(addresses_to_parse):
        raise DataError("Some addresses are empty.")
    if validate_if_any_whitespace_only(addresses_to_parse):
        raise DataError("Some addresses only include whitespace thus cannot be parsed.")
