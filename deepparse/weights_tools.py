from typing import OrderedDict, Union

import torch
from cloudpathlib import CloudPath, S3Path
from torch import nn
from torch.nn import init


def weights_init(m: nn.Module) -> None:
    """
    Function to initialize the weights of a model layers.

    Usage:
        network = Model()
        network.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def handle_weights_upload(
    path_to_model_to_upload: Union[str, S3Path], device: Union[str, torch.device] = "cpu"
) -> OrderedDict:
    if isinstance(path_to_model_to_upload, S3Path):
        # To handle CloudPath path_to_model_weights
        try:
            with path_to_model_to_upload.open("rb") as file:
                checkpoint_weights = torch.load(file, map_location=device)
        except FileNotFoundError as error:
            raise FileNotFoundError("The file in the S3 bucket was not found.") from error
    elif "s3://" in path_to_model_to_upload:
        # To handle str S3-like URI.
        path_to_model_to_upload = CloudPath(path_to_model_to_upload)
        try:
            with path_to_model_to_upload.open("rb") as file:
                checkpoint_weights = torch.load(file, map_location=device)
        except FileNotFoundError as error:
            raise FileNotFoundError("The file in the S3 bucket was not found.") from error
    else:
        # Path is a local one (or a wrongly written S3 URI).
        try:
            checkpoint_weights = torch.load(path_to_model_to_upload, map_location=device)
        except FileNotFoundError as error:
            if "s3" in path_to_model_to_upload or "//" in path_to_model_to_upload or ":" in path_to_model_to_upload:
                raise FileNotFoundError(
                    "Are You trying to use a AWS S3 URI? If so path need to start with s3://."
                ) from error
            raise FileNotFoundError(f"The file {path_to_model_to_upload} was not found.") from error
    return checkpoint_weights
