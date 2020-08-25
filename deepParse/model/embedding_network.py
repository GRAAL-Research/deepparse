from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmbeddingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def __call__(self, input_: torch.Tensor, decomposition_lengths: Tuple[List]):
        device = input_.device
        batch_size = input_.size(0)

        embeddings = torch.zeros(input_.size(1), input_.size(0), input_.size(3) * 2).to(device)

        input_ = input_.transpose(0, 1).float()

        for i in range(input_.size(0)):
            lengths = []

            # reorder decomposition, could use a transpose but take a LOT (like a LOT) of memory
            for decomposition_length in decomposition_lengths:
                lengths.append(decomposition_length[i])
            packed_sequence = pack_padded_sequence(input_[i], torch.tensor(lengths),
                                                   batch_first=True, enforce_sorted=False)

            _, hidden = self.model(packed_sequence, self.hidden)

            encoding = hidden[0].transpose(0, 1)

            word_batch_embedding = torch.zeros(batch_size, 2 * self.hidden_size).to(device)
            for j in range(batch_size):
                word_batch_embedding[j] = encoding[j].reshape(2 * self.hidden_size)

            embeddings[i] = word_batch_embedding

        return embeddings.transpose(0, 1)
