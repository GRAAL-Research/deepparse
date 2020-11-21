import os
import warnings

import requests
import torch
import torch.nn as nn
import torch.nn.init as init

BASE_URL = "https://graal.ift.ulaval.ca/public/deepparse/"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")


def verify_latest_version(model: str) -> bool:
    """
    Verify if the local model is the latest.
    """
    local_model_hash_version = open(os.path.join(CACHE_PATH, model + ".version")).readline()
    download_from_url(model, CACHE_PATH, "version")
    remote_model_hash_version = open(os.path.join(CACHE_PATH, model + ".version")).readline()
    return local_model_hash_version != remote_model_hash_version


def verify_if_model_in_cache(model: str) -> bool:
    """
    Verify if a model is in cache and give warning if not latest.
    """
    try:
        open(os.path.join(CACHE_PATH, model + ".version"))
        if verify_latest_version(model):
            warnings.warn("A newer model is available, you can download it using the download script.")
        return True
    except FileNotFoundError:
        return False


def download_from_url(model: str, saving_dir: str, extension: str):
    """
    Simple function to download the content of a file from a distant repository.
    """
    model_url = BASE_URL + "{}." + extension
    url = model_url.format(model)
    r = requests.get(url)

    os.makedirs(saving_dir, exist_ok=True)

    open(os.path.join(saving_dir, f"{model}.{extension}"), "wb").write(r.content)


def download_weights(model: str, saving_dir: str, verbose: bool = True) -> None:
    """
    Function to download the pre-trained weights of the models.

    Args:
        model: The network type (i.e. fasttext or bpemb).
        saving_dir: The path to the saving directory.
    """
    if verbose:
        print(f"Downloading the weights for the network {model}.")
    download_from_url(model, saving_dir, "ckpt")
    download_from_url(model, saving_dir, "version")


def load_tuple_to_device(padded_address, device):
    """
    Function to load the torch components of a tuple to a device. Since tuple are immutable we return a new tuple with
    the tensor loaded to the device.
    """
    return tuple([element.to(device) if isinstance(element, torch.Tensor) else element for element in padded_address])


def download_data(saving_dir, dataset_name):
    """
    Function to download a dataset (dataset_name) from our public repository and save it into the saving_dir.
    """
    print(f"Download of dataset {dataset_name}")
    dataset_name += ".p"
    url = BASE_URL + dataset_name
    r = requests.get(url)
    os.makedirs(saving_dir, exist_ok=True)

    open(os.path.join(saving_dir, f"{dataset_name}"), 'wb').write(r.content)


def weight_init(m):
    # pylint: disable=too-many-branches, too-many-statements
    """
    Function to initialize the weight of a layer.
    Usage:
        network = Model()
        network.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
