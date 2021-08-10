import math
import os
import warnings

import numpy as np
import poutyne
import requests
import torch

BASE_URL = "https://graal.ift.ulaval.ca/public/deepparse/{}.{}"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")


def latest_version(model: str, cache_path: str) -> bool:
    """
    Verify if the local model is the latest.
    """
    with open(os.path.join(cache_path, model + ".version")) as local_model_hash_file:
        local_model_hash_version = local_model_hash_file.readline()
    download_from_url(model, cache_path, "version")
    with open(os.path.join(cache_path, model + ".version")) as remote_model_hash_file:
        remote_model_hash_version = remote_model_hash_file.readline()
    return local_model_hash_version.strip() == remote_model_hash_version.strip()


def download_from_url(file_name: str, saving_dir: str, file_extension: str):
    """
    Simple function to download the content of a file from a distant repository.
    """
    url = BASE_URL.format(file_name, file_extension)
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()  # raise exception if 404 or other http error
    except requests.exceptions.ConnectTimeout:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) "
            "Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)",
            "Referer": "http://example.com"
        }
        r = requests.get(url, timeout=5, headers=headers)
        r.raise_for_status()  # raise exception if 404 or other http error
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


def load_tuple_to_device(padded_address, device):
    # pylint: disable=consider-using-generator
    """
    Function to load the torch components of a tuple to a device. Since tuple are immutable we return a new tuple with
    the tensor loaded to the device.
    """
    return tuple([element.to(device) if isinstance(element, torch.Tensor) else element for element in padded_address])


def handle_poutyne_version() -> float:
    """
    Handle the retrieval of the major and minor part of the Poutyne version
    """
    full_version = poutyne.version.__version__
    components_parts = full_version.split(".")
    major = components_parts[0]
    minor = components_parts[1]
    version = f"{major}.{minor}"
    return float(version)


def valid_poutyne_version():
    """
    Validate Poutyne version is greater than 1.2 for using a str checkpoint. Version before does not support that
    feature.
    """
    return handle_poutyne_version() >= 1.2


def handle_pre_trained_checkpoint(model_type_checkpoint: str) -> str:
    """
    Handle the checkpoint formatting for pre trained models.
    """
    if not valid_poutyne_version():
        raise NotImplementedError(
            f"To load the pre-trained {model_type_checkpoint} model, you need to have a Poutyne version"
            "greater than 1.1 (>1.1)")
    model_path = os.path.join(CACHE_PATH, f"{model_type_checkpoint}.ckpt")

    if not os.path.isfile(model_path):
        download_weights(model_type_checkpoint, CACHE_PATH, verbose=True)
    elif not latest_version(model_type_checkpoint, cache_path=CACHE_PATH):
        warnings.warn("A newer model of fasttext is available, you can download it using the download script.",
                      UserWarning)
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
            raise NotImplementedError("To load a string path to a model, you need to have a Poutyne version"
                                      "greater than 1.1 (>1.1)")
    else:
        raise ValueError("The checkpoint is not valid. Can be a path in a string format (e.g. 'a_path_.ckpt'), "
                         "'fasttext' or 'bpemb'.")

    return checkpoint


def indices_splitting(num_data: int, train_ratio: float, seed: int = 42):
    """
    Split indices into train and valid
    """
    np.random.seed(seed)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = math.floor(train_ratio * num_data)

    train_indices = indices[:split]
    valid_indices = indices[split:]

    return train_indices, valid_indices
