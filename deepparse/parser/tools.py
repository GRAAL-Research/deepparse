import math
import os
from typing import List, Tuple

import numpy as np
import torch


def validate_if_new_prediction_tags(checkpoint_weights: dict) -> bool:
    return checkpoint_weights.get("prediction_tags") is not None


def validate_if_new_seq2seq_params(checkpoint_weights: dict) -> bool:
    return checkpoint_weights.get("seq2seq_params") is not None


def pretrained_parser_in_directory(logging_path: str) -> bool:
    # We verify if an address retrained address parser is in the directory
    files_in_directory = get_files_in_directory(logging_path)

    return len(get_address_parser_in_directory(files_in_directory)) > 0


def get_files_in_directory(logging_path: str) -> List[str]:
    files_path_with_directories = [files for root, dirs, files in os.walk(os.path.abspath(logging_path))]

    # We unwrap the list of list
    files_in_directory = [elem for sublist in files_path_with_directories for elem in sublist]
    return files_in_directory


def get_address_parser_in_directory(files_in_directory: List[str]) -> List:
    return [
        file_name for file_name in files_in_directory if "_address_parser" in file_name and "retrained" in file_name
    ]


def load_tuple_to_device(padded_address: Tuple, device: torch.device):
    # pylint: disable=consider-using-generator
    """
    Function to load the torch components of a tuple to a device. Since tuple are immutable we return a new tuple with
    the tensor loaded to the device.
    """
    return tuple([element.to(device) if isinstance(element, torch.Tensor) else element for element in padded_address])


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
