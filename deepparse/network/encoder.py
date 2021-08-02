# temporary fix for _forward_unimplemented for torch 1.6 https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=W0223

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..weights_init import weights_init


class Encoder(nn.Module):
    """
    Encoder module that use a LSTM to encode a sequence.

    Args:
        input_size (int): The input size of the encoder.
        hidden_size (int): The hidden size of the encoder.
        num_layers (int): The number of layer to the encoder.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm.apply(weights_init)

    def forward(self, to_predict: torch.Tensor, lengths_tensor: torch.Tensor) -> Tuple:
        """
            Callable method to encode the components of an address.

            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                lengths_tensor (~torch.Tensor): The lengths of the batch elements (since packed).

            Return:
                A tuple of the address components encoding.
        """

        packed_sequence = pack_padded_sequence(to_predict, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)

        _, hidden = self.lstm(packed_sequence)

        return hidden
