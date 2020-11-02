# temporary fix for _forward_unimplemented for torch 1.6 https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=W0223
from typing import Tuple

import torch
import torch.nn as nn

from ..tools import weight_init


class Decoder(nn.Module):
    """
    Decoder module that use a LSTM to decode a previously encoded sequence and a linear layer to map
    the decoded sequence tags.

    Args:
        input_size (int): The input size of the decoder.
        hidden_size (int): The hidden size of the decoder.
        num_layers (int): The number of layer to the decoder.
        output_size (int): The output size of the decoder (i.e. the number of tags to predict on).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.lstm.apply(weight_init)

        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.apply(weight_init)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, to_predict: torch.Tensor, hidden: torch.Tensor) -> Tuple:
        """
            Callable method to decode the components of an address.

            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                hidden (~torch.Tensor): The hidden state of the decoder.

            Return:
                A tuple (``x``, ``y``) where ``x`` is the address components tags predictions and y is the hidden
                states.

        """
        output, hidden = self.lstm(to_predict.float(), hidden)

        output_prob = self.softmax(self.linear(output[0]))

        return output_prob, hidden
