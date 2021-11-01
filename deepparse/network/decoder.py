# temporary fix for _forward_unimplemented for torch 1.6 https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=W0223
from typing import Tuple

import torch
import torch.nn as nn

from ..weights_init import weights_init


class Decoder(nn.Module):
    """
    Decoder module that use a LSTM to decode a previously encoded sequence and a linear layer to map
    the decoded sequence tags.

    Args:
        input_size (int): The input size of the decoder.
        hidden_size (int): The hidden size of the decoder.
        num_layers (int): The number of layer to the decoder.
        output_size (int): The output size of the decoder (i.e. the number of tags to predict on).
        attention_mechanism (bool): Either or not to use attention mechanism in forward pass.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int,
                 attention_mechanism: bool) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.lstm.apply(weights_init)

        self.linear_layer_set_up(output_size, hidden_size=hidden_size)

        self.softmax = nn.LogSoftmax(dim=1)

        forward_function = self._forward
        if attention_mechanism:
            forward_function = self._forward_attention_mechanism

        self.forward = forward_function

    def _forward(self, to_predict: torch.Tensor, hidden: torch.Tensor) -> Tuple:
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

    def _forward_attention_mechanism(self, to_predict: torch.Tensor, hidden: torch.Tensor,
                                     encoder_outputs: torch.Tensor, lengths: torch.Tensor) -> Tuple:
        """
            Callable method to decode the components of an address using attention mechanism.

            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                hidden (~torch.Tensor): The hidden state of the decoder.
                encoder_outputs (~torch.Tensor): The encoder outputs for the attention mechanism weighs if needed.
                lengths (~torch.Tensor) : The lengths of the batch elements (since packed).

            Return:
                A tuple (``x``, ``y``, ``z``) where ``x`` is the address components tags predictions, y is the hidden
                states and `̀`z`` is the attention weights.

        """
        output, hidden = self.lstm(to_predict.float(), hidden)

        output_prob = self.softmax(self.linear(output[0]))

        return output_prob, hidden

    def linear_layer_set_up(self, output_size: int, hidden_size: int = 1024):
        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.apply(weights_init)
