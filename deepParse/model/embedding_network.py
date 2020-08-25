from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmbeddingNetwork(nn.Module):
    """
    Embedding Network to represent the address components byte-pair embedding representation using a LSTM.

    Args:
        input_size (int): The input size of the LSTM.
        hidden_size (int): The hidden size of the LSTM.
        num_layers (int): The number of layer of the LSTM. Default is one (1) layer.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.model = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def __call__(self, to_predict: torch.Tensor, decomposition_lengths: Tuple[List]) -> torch.Tensor:
        """
            Callable method to .

            Args:
                to_predict (~torch.Tensor): The address to extract the embedding on.
                decomposition_lengths (Tuple[List]) : The decomposition lengths of the address components.

            Return:
                The embedded address vector tensor.

        """
        device = to_predict.device
        batch_size = to_predict.size(0)

        embeddings = torch.zeros(to_predict.size(1), to_predict.size(0), to_predict.size(3) * 2).to(device)

        to_predict = to_predict.transpose(0, 1).float()

        for i in range(to_predict.size(0)):
            lengths = []

            # reorder decomposition, could use a transpose but take a LOT (like a LOT) of memory
            for decomposition_length in decomposition_lengths:
                lengths.append(decomposition_length[i])

            packed_sequence = pack_padded_sequence(to_predict[i], torch.tensor(lengths),
                                                   batch_first=True, enforce_sorted=False)

            _, hidden = self.model(packed_sequence, self.hidden)
            encoding = hidden[0].transpose(0, 1)

            # filling the embedding by idx
            word_batch_embedding = torch.zeros(batch_size, 2 * self.hidden_size).to(device)
            for j in range(batch_size):
                word_batch_embedding[j] = encoding[j].reshape(2 * self.hidden_size)

            embeddings[i] = word_batch_embedding

        return embeddings.transpose(0, 1)

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.model.eval()
