# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# temporary fix for _forward_unimplemented for PyTorch 1.6 https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=W0223

from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbeddingNetwork(nn.Module):
    """
    Embedding Network to represent the address components byte-pair embedding representation using a LSTM.

    Args:
        input_size (int): The input size of the LSTM.
        hidden_size (int): The hidden size of the LSTM.
        num_layers (int): The number of layer of the LSTM. Default is one (1) layer.
        maxpool (bool): Either or not to add a maximum pooling layer after the embedding composition. Default is false.
        maxpool_kernel_size (int): The kernel size of the maximum pooling layer. Default is three (3).
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 projection_size: int,
                 num_layers: int = 1,
                 maxpool=False,
                 maxpool_kernel_size=3) -> None:
        # pylint: disable=too-many-arguments
        super().__init__()

        self.hidden_size = hidden_size
        self.model = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        self.projection_layer = nn.Linear(2 * hidden_size, projection_size)

        self.maxpool_kernel_size = maxpool_kernel_size if maxpool else 1
        self.maxpooling_layer = nn.MaxPool1d(maxpool_kernel_size) if maxpool else None

    def forward(self, to_predict: torch.Tensor, decomposition_lengths: Tuple[List]) -> torch.Tensor:
        # pylint: disable=too-many-locals
        """
        Callable method to aggregate the byte-pair embeddings from decomposed words.

        Args:
            to_predict (~torch.Tensor): The address to extract the embedding on.
            decomposition_lengths (tuple[list]) : The decomposition lengths of the address components.

        Return:
            The embedded address vector tensor.
        """
        device = to_predict.device
        batch_size = to_predict.size(0)

        embeddings = torch.zeros(to_predict.size(1), to_predict.size(0),
                                 int(to_predict.size(3) / self.maxpool_kernel_size)).to(device)

        to_predict = to_predict.transpose(0, 1).float()

        for i in range(to_predict.size(0)):
            lengths = []

            # reorder decomposition, could use a transpose but take a LOT (like a LOT) of memory
            for decomposition_length in decomposition_lengths:
                lengths.append(decomposition_length[i])

            packed_sequence = pack_padded_sequence(to_predict[i],
                                                   torch.tensor(lengths).cpu(),
                                                   batch_first=True,
                                                   enforce_sorted=False)

            packed_output, _ = self.model(packed_sequence)

            # pad packed the output to be applied later on in the projection layer
            padded_output, padded_output_lengths = pad_packed_sequence(packed_output, batch_first=True)

            # filling the embedding by idx
            word_context = torch.zeros(padded_output.size(0), padded_output.size(2)).to(device)
            for j in range(batch_size):
                word_context[j] = padded_output[j, padded_output_lengths[j] - 1, :]

            # projection layer from dim 600 to 300
            projection_output = self.projection_layer(word_context)

            if self.maxpooling_layer is not None:
                projection_output = self._max_pool(projection_output)

            embeddings[i] = projection_output

        return embeddings.transpose(0, 1)

    def _max_pool(self, projection_output):
        """
        Max pooling the projection output of the projection layer.
        """
        pooled_output = self.maxpooling_layer(
            projection_output.view(1, projection_output.size(0), projection_output.size(1)))
        projection_output = pooled_output.view(pooled_output.size(1), pooled_output.size(2))

        return projection_output
