from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..tools import weight_init


class Encoder(nn.Module):
    """
    Encoder module that use a LSTM to encode a sequence.

    Args:
        input_size (int): The input size of the decoder.
        hidden_size (int): The hidden size of the decoder.
        num_layers (int): The number of layer to the decoder.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm.apply(weight_init)

    def forward(self, to_predict: torch.Tensor, lengths_tensor: torch.Tensor) -> Tuple:
        """
            Callable method to encode the components of an address.

            rgs:
                to_predict (~torch.Tensor): The elements to predict the tags.
                lengths_tensor (~torch.Tensor): The lengths of the batch elements (since packed).

            Return:
                A tuple of the address components encoding.
        """

        packed_sequence = pack_padded_sequence(to_predict, lengths_tensor, batch_first=True)

        _, hidden = self.lstm(packed_sequence, self.hidden)

        return hidden

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.lstm.eval()
