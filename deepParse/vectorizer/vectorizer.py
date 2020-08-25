from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from deepParse.embeddings_model.embeddings_model import EmbeddingsModel


class Vectorizer(ABC):
    """
    Vectorizer abstract class to vectorize an address into a [embeddings]

    Args:
        embeddings_model (~deepParse.embeddings_model.EmbeddingsModel): A callable embeddings model.
        eos_token (int): The end of sentence token to use.

    """

    def __init__(self, embeddings_model: EmbeddingsModel, eos_token: int) -> None:
        self.embeddings_model = embeddings_model
        self.eos_token = eos_token

    @abstractmethod
    def __call__(self, addresses):
        """
        Method to vectorizer addresses.
        Args:
            addresses (List[str]): The addresses to vectorize.

        Return:
            The addresses elements (components) embeddings vector.
        """

        pass


class FastTextVectorizer(Vectorizer):
    """
    FastText vectorizer to convert an address into fastText embeddings.
    """

    def __call__(self, addresses: List[str]) -> List:
        batch = []

        for address in addresses:
            batch.append(self._vectorize_sequence(address))

        return batch

    def _vectorize_sequence(self, address: str) -> List:
        """
        Method to vectorize the address

        Args:
            address (str): Address to vectorize using fastText.

        Return:
            A list of word vector.
        """

        input_sequence = []

        for word in address.split():
            embedding = self.embeddings_model(word)
            input_sequence.append(embedding)
        return input_sequence


class BPEmbVectorizer(Vectorizer):
    """
    BPEmb vectorizer to convert an address into BPEmb embeddings.
    """

    def __call__(self, addresses: List[str]) -> List[Tuple]:
        batch = []
        self._max_length = 0

        for address in addresses:
            input_sequence, word_decomposition_lengths = self._vectorize_sequence(address)

            batch.append((input_sequence, word_decomposition_lengths))

        # todo in a method
        for decomposed_sequence, _, _ in batch:
            for decomposition in decomposed_sequence:
                if len(decomposition) != self._max_length:
                    for i in range(self._max_length - len(decomposition)):
                        decomposition.append(
                            np.ones(self.embeddings_model.dim) * 0)  # todo validate if the dim is ok

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
