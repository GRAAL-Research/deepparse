from typing import List, Tuple

import numpy as np

from .vectorizer import Vectorizer
from ..embeddings_model.embeddings_model import EmbeddingsModel


class BPEmbVectorizer(Vectorizer):
    """
    BPEmb vectorizer to convert an address into BPEmb embeddings.

    Args:
        embeddings_model (~deepParse.embeddings_model.EmbeddingsModel): A callable embeddings model.

    """

    def __init__(self, embeddings_model: EmbeddingsModel):
        super().__init__(embeddings_model)

        self.padding_value = 0

    def __call__(self, addresses: List[str]) -> List[Tuple]:
        """
        Method to vectorizer addresses.

        Args:
            addresses (List[str]): The addresses to vectorize.

        Return:
            A tuple of the addresses elements (components) embeddings vector and the word decomposition lengths.
        """

        batch = []
        self._max_length = 0

        for address in addresses:
            input_sequence, word_decomposition_lengths = self._vectorize_sequence(address)

            batch.append((input_sequence, word_decomposition_lengths))

        self._decomposed_sequence_padding(batch)

        return batch

    def _vectorize_sequence(self, address: str) -> Tuple[List, List]:
        """
        Method to vectorize the address

        Args:
            address (str): Address to vectorize using BPEmb.

        Return:
            A tuple of list of word vector and the word decomposition lengths.
        """
        input_sequence = []
        word_decomposition_lengths = []

        for word in address.split():
            word_decomposition = []
            bpe_decomposition = self.embeddings_model(word)
            word_decomposition_lengths.append(len(bpe_decomposition))
            for i in range(bpe_decomposition.shape[0]):
                word_decomposition.append(bpe_decomposition[i])
            input_sequence.append(word_decomposition)

        for decomposition in input_sequence:
            if len(decomposition) > self._max_length:
                self._max_length = len(decomposition)

        return input_sequence, word_decomposition_lengths

    def _decomposed_sequence_padding(self, batch: List[Tuple]) -> None:
        """
        Method to add padding to the decomposed sequence
        """
        for decomposed_sequence, _ in batch:
            for decomposition in decomposed_sequence:
                if len(decomposition) != self._max_length:
                    for i in range(self._max_length - len(decomposition)):
                        decomposition.append(
                            np.ones(
                                self.embeddings_model.dim) * self.padding_value)  # todo validate property is working
