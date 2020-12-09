import os
import warnings

import requests
import torch
import torch.nn as nn
import torch.nn.init as init

BASE_URL = "https://graal.ift.ulaval.ca/public/deepparse/"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")


def latest_version(model: str, cache_path: str) -> bool:
    """
    Verify if the local model is the latest.
    """
    local_model_hash_file = open(os.path.join(cache_path, model + ".version"))
    local_model_hash_version = local_model_hash_file.readline()
    local_model_hash_file.close()
    download_from_url(model, cache_path, "version")
    remote_model_hash_file = open(os.path.join(cache_path, model + ".version"))
    remote_model_hash_version = remote_model_hash_file.readline()
    remote_model_hash_file.close()
    return local_model_hash_version.strip() == remote_model_hash_version.strip()


def download_from_url(file_name: str, saving_dir: str, file_extension: str):
    """
    Simple function to download the content of a file from a distant repository.
    """
    model_url = BASE_URL + "{}." + file_extension
    url = model_url.format(file_name)
    r = requests.get(url)
    r.raise_for_status()  # raise exception if 404 or other http error

    os.makedirs(saving_dir, exist_ok=True)

    file = open(os.path.join(saving_dir, f"{file_name}.{file_extension}"), "wb")
    file.write(r.content)
    file.close()


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
    """
    Function to load the torch components of a tuple to a device. Since tuple are immutable we return a new tuple with
    the tensor loaded to the device.
    """
    return tuple([element.to(device) if isinstance(element, torch.Tensor) else element for element in padded_address])


def handle_checkpoint(checkpoint):
    """
    Handle the checkpoint format validity and path.
    """
    if checkpoint in ("best", "last"):
        pass
    elif isinstance(checkpoint, int):
        pass
    elif checkpoint == "fasttext":
        if not latest_version("fasttext", cache_path=CACHE_PATH):
            warnings.warn("A newer model of fasttext is available, you can download it using the download script.")
        checkpoint = os.path.join(CACHE_PATH, "fasttext.p")
    elif checkpoint == "bpemb":
        if not latest_version("bpemb", cache_path=CACHE_PATH):
            warnings.warn("A newer model bpemb is available, you can download it using the download script.")
        checkpoint = os.path.join(CACHE_PATH, "bpemb.p")
    elif isinstance(checkpoint, str) and checkpoint.endswith(".p"):
        pass
    else:
        raise ValueError("The checkpoint is not valid. Can be 'best', 'last', a int, a path in a string format, "
                         "'fasttext' or 'bpemb'.")

    return checkpoint


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
