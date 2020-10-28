from typing import List, Tuple

import numpy as np

from .vectorizer import Vectorizer
from ..embeddings_models.embeddings_model import EmbeddingsModel


class BPEmbVectorizer(Vectorizer):
    """
    BPEmb vectorizer to convert an address into BPEmb embedding where each word is decomposed into subword units that
    are in turn embedded as a vector
    """

    def __init__(self, embeddings_model: EmbeddingsModel) -> None:
        super().__init__(embeddings_model)

        self.padding_value = 0

    def __call__(self, addresses: List[str]) -> List[Tuple]:
        """
        Method to vectorizer addresses.

        Args:
            addresses (list[str]): The addresses to vectorize.

        Return:
            A tuple of the addresses elements (components) embedding vectosr and the word decomposition lengths.
        """
        self._max_length = 0
        batch = [self._vectorize_sequence(address) for address in addresses]
        self._decomposed_sequence_padding(batch)
        return batch

    def _vectorize_sequence(self, address: str) -> Tuple[List, List]:
        """
        Method to vectorize the address.

        Args:
            address (str): Address to vectorize using BPEmb.

        Return:
            A tuple of list of word vector and the word decomposition lengths.
        """
        input_sequence = []
        word_decomposition_lengths = []
        address = address.replace(",", "")  # see issue 56 https://github.com/GRAAL-Research/deepparse/issues/56
        for word in address.split():
            bpe_decomposition = self.embeddings_model(word)
            word_decomposition_lengths.append(len(bpe_decomposition))
            input_sequence.append(list(bpe_decomposition))

        self._max_length = max(self._max_length, max(word_decomposition_lengths))

        return input_sequence, word_decomposition_lengths

    def _decomposed_sequence_padding(self, batch: List[Tuple]) -> None:
        """
        Method to add padding to the decomposed sequence.
        """
        for decomposed_sequence, _ in batch:
            for decomposition in decomposed_sequence:
                if len(decomposition) != self._max_length:
                    decomposition.extend([np.ones(self.embeddings_model.dim) * [self.padding_value]] *
                                         (self._max_length - len(decomposition)))
