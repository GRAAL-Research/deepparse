import math
import os
from typing import List, Tuple, OrderedDict

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
    Function to load the torch components of a tuple to a device. Since tuples are immutable, we return a new tuple with
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


def handle_model_name(model_type: str, attention_mechanism: bool) -> Tuple[str, str]:
    """
    Handle the model type name matching with proper seq2seq model type name.
    Args:
        model_type (str): The type of the model.
        attention_mechanism (bool): Either or not, the model uses an attention mechanism.

    Return:
        A tuple of two strings where the first element is the model_type and the second is the formatted name.
    """
    model_type = model_type.lower()

    # To handle retrained model using attention mechanism.
    if 'attention' in model_type:
        if not attention_mechanism:
            raise ValueError(
                f"Model-type {model_type} requires attention mechanism. " f"Set attention_mechanism to True."
            )
        model_type = model_type.replace('attention', '')

    if model_type in ("lightest", "fasttext-light"):
        model_type = "fasttext-light"  # We change name to 'fasttext-light' since lightest = fasttext-light
        formatted_name = "FastTextLight"
    elif model_type in ("fastest", "fasttext"):
        model_type = "fasttext"  # We change name to fasttext since fastest = fasttext
        formatted_name = "FastText"
    elif model_type in ("best", "bpemb"):
        model_type = "bpemb"  # We change name to bpemb since best = bpemb
        formatted_name = "BPEmb"
    else:
        raise ValueError(
            f"Could not handle {model_type}. Read the docs at https://deepparse.org/ for possible model types."
        )

    if attention_mechanism:
        model_type += "Attention"
        formatted_name += "Attention"
    return model_type, formatted_name


def infer_model_type(checkpoint_weights: OrderedDict, attention_mechanism: bool) -> (str, bool):
    """
    Function to infer the model type using the weights matrix.
    We first try to use the "model_type" key added by our retrain process.
    If this fails, we infer it using our knowledge of the layers' names.
    For example, BPEmb model uses an embedding network, thus, if `embedding_network.model.weight_ih_l0` is present,
    we can say that it is such a type; otherwise, it is a FastText model.
    Finally, to handle the attention model, we use a similar approach but using the
    `decoder.linear_attention_mechanism_encoder_outputs.weight` layer name to deduct the presence of
    attention mechanism.

    Args:
        checkpoint_weights (OrderedDict): The weights matrix.
        attention_mechanism (bool): Either or not the model uses an attention mechanism or not.

    Return:
        A tuple where the first element is the model_type name and the second element is the attention_mechanism flag.

    """
    inferred_model_type = checkpoint_weights.get("model_type")
    if inferred_model_type is not None:
        model_type = inferred_model_type
    else:
        if "embedding_network.model.weight_ih_l0" in checkpoint_weights.keys():
            model_type = "bpemb"
        else:
            model_type = "fasttext"

    if "decoder.linear_attention_mechanism_encoder_outputs.weight" in checkpoint_weights.keys():
        attention_mechanism = True

    return model_type, attention_mechanism
