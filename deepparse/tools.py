import os

import requests
import torch
import torch.nn as nn
import torch.nn.init as init

base_url = "https://davebulaval.github.io/deepparse-external-assets/"


def verify_latest_version(model_type: str, root_path: str) -> bool:
    """
    Verify if the local model is the latest.
    """
    local_model_hash_version = open(os.path.join(root_path, model_type + ".version")).readline()
    download_from_url(model_type, root_path, 'version')
    remote_model_hash_version = open(os.path.join(root_path, model_type + ".version")).readline()
    return local_model_hash_version != remote_model_hash_version


def download_from_url(model_type: str, saving_dir: str, data_type: str):
    """
    Simple function to download the content of a file from a distant repository.
    """
    model_url = base_url + "{}." + data_type
    url = model_url.format(model_type)
    r = requests.get(url)

    os.makedirs(saving_dir, exist_ok=True)

    open(os.path.join(saving_dir, f"{model_type}.{data_type}"), 'wb').write(r.content)


def download_weights(model_type: str, saving_dir: str) -> None:
    """
    Function to download the pre-trained weights of the models.

    Args:
        model_type: The network type (i.e. fasttext or bpemb).
        saving_dir: The path to the saving directory.
    """
    print(f"Downloading the weights for the network {model_type}.")
    download_from_url(model_type, saving_dir, 'ckpt')
    download_from_url(model_type, saving_dir, 'version')


def load_tuple_to_device(padded_address, device):
    """
    Function to load the torch components of a tuple to a device. Since tuple are immutable we return a new tuple with
    the tensor loaded to the device.
    """
    return tuple([element.to(device) if isinstance(element, torch.Tensor) else element for element in padded_address])


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
