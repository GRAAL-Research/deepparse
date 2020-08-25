from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbeddingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, num_layers=1, maxpool=False,
                 maxpool_kernel_size=3):
        super().__init__()

        self.model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        self.projection_layer = nn.Linear(2 * hidden_size, projection_size)

        self.maxpool_kernel_size = maxpool_kernel_size if maxpool else 1
        self.maxpooling_layer = nn.MaxPool1d(maxpool_kernel_size) if maxpool else None

    def forward(self, input_: torch.Tensor, decomposition_lengths: Tuple[List]):
        device = input_.device

        embeddings = torch.zeros(input_.size(1), input_.size(0), int(input_.size(3) / self.maxpool_kernel_size)).to(
            device)

        input_ = input_.transpose(0, 1).float()

        for i in range(input_.size(0)):
            lengths = []
            for decomposition_length in decomposition_lengths:
                lengths.append(decomposition_length[i])
            packed_sequence = pack_padded_sequence(input_[i], torch.tensor(lengths),
                                                   batch_first=True, enforce_sorted=False)

            packed_output, hidden = self.model(packed_sequence, self.hidden)

            padded_output, padded_output_lengths = pad_packed_sequence(packed_output, batch_first=True,
                                                                       padding_value=-100)

            word_context = torch.zeros(padded_output.size(0), padded_output.size(2)).to(self.device)
            for j in range(padded_output_lengths.size(0)):
                word_context[j] = padded_output[j, padded_output_lengths[j] - 1, :]

            projection_output = self.projection_layer(word_context)

            if self.maxpooling_layer is not None:
                pooled_output = self.maxpooling_layer(
                    projection_output.view(1, projection_output.size(0), projection_output.size(1)))
                projection_output = pooled_output.view(pooled_output.size(1), pooled_output.size(2))

            embeddings[i] = projection_output

        return embeddings.transpose(0, 1)
