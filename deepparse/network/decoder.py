# temporary fix for _forward_unimplemented for torch 1.6 https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=W0223, too-many-arguments
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
        self.attention_mechanism = attention_mechanism
        if attention_mechanism:
            # Since layer also have attention mechanism
            self.hidden_size = hidden_size
            input_size = input_size + hidden_size
            self.attention_mechanism_set_up()
        self.softmax = nn.LogSoftmax(dim=1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.lstm.apply(weights_init)

        self.linear_layer_set_up(output_size, hidden_size=hidden_size)

    def forward(self, to_predict: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                lengths: torch.Tensor) -> Tuple:
        """
            Callable method to decode the components of an address using attention mechanism.

            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                hidden (~torch.Tensor): The hidden state of the decoder.
                encoder_outputs (~torch.Tensor): The encoder outputs for the attention mechanism weighs if needed.
                lengths (~torch.Tensor) : The lengths of the batch elements (since packed).

            Return:
                A tuple (``x``, ``y``, ``z``) where ``x`` is the address components tags predictions, y is the hidden
                states and `̀`z`` is None if no attention mechanism is setter or the attention weights.

        """
        to_predict = to_predict.float()
        attention_weights = None
        if self.attention_mechanism:
            to_predict, attention_weights = self._attention_mechanism_forward(to_predict, hidden, encoder_outputs,
                                                                              lengths)

        output, hidden = self.lstm(to_predict, hidden)

        output_prob = self.softmax(self.linear(output[0]))

        return output_prob, hidden, attention_weights  # attention_weights: None or the real attention weights

    def linear_layer_set_up(self, output_size: int, hidden_size: int = 1024):
        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.apply(weights_init)

    def attention_mechanism_set_up(self, hidden_size: int = 1024):
        self.linear_attention_mechanism_encoder_outputs = nn.Linear(hidden_size, hidden_size)
        self.linear_attention_mechanism_encoder_outputs.apply(weights_init)

        self.linear_attention_mechanism_previous_hidden = nn.Linear(hidden_size, hidden_size)
        self.linear_attention_mechanism_previous_hidden.apply(weights_init)

        self.weights = nn.Parameter(torch.ones(1, hidden_size))

    def _attention_mechanism_forward(self, to_predict: torch.Tensor, hidden: torch.Tensor,
                                     encoder_outputs: torch.Tensor, lengths: torch.Tensor) -> Tuple:
        """
        Compute the attention mechanism weights and context vector
        Return:
            A tuple (``x``, ``y``) where ``x`` is the to_predict vector with the context vector and y is the attention
            weights.
        """
        unweighted_alignments = torch.tanh(
            self.linear_attention_mechanism_encoder_outputs(encoder_outputs) +
            self.linear_attention_mechanism_previous_hidden(hidden[0].transpose(0, 1)))
        alignments_scores = torch.matmul(self.weights.view(1, 1, self.hidden_size),
                                         unweighted_alignments.transpose(1, 2))

        max_length = lengths.max().item()
        mask = torch.arange(max_length)[None, :] < lengths[:, None].to(
            "cpu")  # We switch the lengths to cpu for the comparison
        mask = mask.unsqueeze(1)
        alignments_scores[~mask] = float("-inf")

        attention_weights = nn.functional.softmax(alignments_scores, dim=2)

        context_vector = torch.matmul(attention_weights, encoder_outputs)

        attention_input = torch.cat((to_predict, context_vector.transpose(0, 1)), 2)
        return attention_input, attention_weights
