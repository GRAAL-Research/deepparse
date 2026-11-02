# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from typing import List, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbeddingNetwork(nn.Module):
    """
    Embedding Network to represent the address components byte-pair embedding representation using an LSTM.

    Args:
        input_size (int): The input size of the LSTM.
        hidden_size (int): The hidden size of the LSTM.
        num_layers (int): The number of layers of the LSTM. The default value is ``1``, namely one layer.
        maxpool (bool): Either or not to add a maximum pooling layer after the embedding composition. The default
            value is ``False``.
        maxpool_kernel_size (int): The kernel size of the maximum pooling layer. The default value is ``3``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        projection_size: int,
        num_layers: int = 1,
        maxpool: bool = False,
        maxpool_kernel_size: int = 3,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__()

        self.hidden_size = hidden_size
        self.model = nn.LSTM(
            input_size,
            self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.projection_layer = nn.Linear(2 * hidden_size, projection_size)

        self.maxpool_kernel_size = maxpool_kernel_size if maxpool else 1
        self.maxpooling_layer = nn.MaxPool1d(maxpool_kernel_size) if maxpool else None

    def forward(self, to_predict: torch.Tensor, decomposition_lengths: Tuple[List]) -> torch.Tensor:
        # pylint: disable=too-many-locals
        """
        Callable method to aggregate the byte-pair embeddings from decomposed words.

        Args:
            to_predict (~torch.Tensor): The address to extract the embedding on, of shape (batch size, number of
                words, number of sub-words of the longest word, embeddings dimension).
            decomposition_lengths (tuple[list]) : The decomposition lengths of the address components (the number of
                sub-words of each word, at least ``1`` for the padding words).

        Return:
            The embedded address vector tensor.
        """
        batch_size, sequence_length, max_decomposition_length, embeddings_dimension = to_predict.size()

        # Each word of each address is an independent sequence of sub-words for the LSTM, so all the words of the
        # batch go through the LSTM in a single call (one call per word position, with Python loops, made the BPEmb
        # models CPU-bound).
        words = to_predict.reshape(batch_size * sequence_length, max_decomposition_length, embeddings_dimension).float()
        words_lengths = torch.tensor(
            [length for address_lengths in decomposition_lengths for length in address_lengths[:sequence_length]],
            dtype=torch.int64,
        )

        packed_sequence = pack_padded_sequence(words, words_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.model(packed_sequence)

        # Pad packed the output to take the output of the last sub-word of each word.
        padded_output, padded_output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        words_indices = torch.arange(padded_output.size(0), device=padded_output.device)
        word_context = padded_output[words_indices, padded_output_lengths.to(padded_output.device) - 1]

        # Projection layer from dim 600 to 300.
        projection_output = self.projection_layer(word_context)

        if self.maxpooling_layer is not None:
            projection_output = self._max_pool(projection_output)

        return projection_output.view(batch_size, sequence_length, -1)

    def _max_pool(self, projection_output: torch.Tensor) -> torch.Tensor:
        """
        Max pooling the projection output of the projection layer.
        """
        pooled_output = self.maxpooling_layer(
            projection_output.view(1, projection_output.size(0), projection_output.size(1))
        )
        projection_output = pooled_output.view(pooled_output.size(1), pooled_output.size(2))

        return projection_output
