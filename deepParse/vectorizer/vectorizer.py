from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from deepParse.converter.converter import TargetConverter
from deepParse.embeddings_model.embeddings_model import EmbeddingsModel


class Vectorizer(ABC):
    """
    Vectorizer abstract class to vectorize a pair (Address, [Target]) into a pair ([embeddings], [tags idx])

    Args:
        embeddings_model (~deepParse.embeddings_model.EmbeddingsModel): A callable embeddings model.
        target_converter (~deepParse.converter.TargetConverter): A target converter that convert from a string to a
            predefine idx.
        eos_token (int): The end of sentence token to use.

    """

    def __init__(self, embeddings_model: EmbeddingsModel, target_converter: TargetConverter, eos_token: int) -> None:
        self.embeddings_model = embeddings_model
        self.target_converter = target_converter
        self.eos_token = eos_token

    @abstractmethod
    def __call__(self, pairs_batch):
        pass

    def _convert_target(self, target_tags: List) -> List:
        """
        Method to convert the target tags into a target idx.

        Args:
            target_tags (List): A list of string tags to be converted.

        Return:
            A list of idx associated with the target tags.
        """
        target_sequence = []
        for target_tag in target_tags:
            target_sequence.append(self.target_converter(target_tag))

        target_sequence.append(self.eos_token)
        return target_sequence


class FastTextVectorizer(Vectorizer):
    """
    FastText vectorizer to convert a pair (Address, [Target]) into a pair ([embeddings], [tags idx])

    Args:
        embeddings_model (~deepParse.embeddings_model.EmbeddingsModel): A callable embeddings model.
        target_converter (~deepParse.converter.TargetConverter): A target converter that convert from a string to a
            predefine idx.
        eos_token (int): The end of sentence token to use. Default is 8.

    """

    def __init__(self, embeddings_model: EmbeddingsModel, target_converter: TargetConverter, eos_token: int = 8):
        super().__init__(embeddings_model, target_converter, eos_token)

    def __call__(self, pairs_batch):  # todo format of input and output
        """
        Method to call the vectorizer over a pairs of batch elements where the first one is the address and the second
        is a list of target tags.
        Args:
            pairs_batch (): The paired elements to vectorize.

        Return:
            The sorted element converter into either embeddings vector or target idx.
        """
        batch = []

        for pair in pairs_batch:
            input_sequence = self._vectorize_sequence(pair[0])

            target_sequence = self._convert_target(pair[1])

            batch.append((input_sequence, target_sequence))

        return sorted(batch, key=lambda x: len(x[0]), reverse=True)  # @Marouane pourquoi on sort ici les elements ?
                                                                        # On les sort car la fonction de transformation en tenseurs (ToTensor) prend un liste ordonnée
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
    BPEmb vectorizer to convert a pair (Address, [Target]) into a pair ([embeddings], [tags idx])

    Args:
        embeddings_model (~deepParse.embeddings_model.EmbeddingsModel): A callable embeddings model.
        target_converter (~deepParse.converter.TargetConverter): A target converter that convert from a string to a
            predefine idx.
        eos_token (int): The end of sentence token to use. Default is 8.
        padding_value (int): The padding value to use for padding the sequence. Default is -100.

    """

    def __init__(self, embeddings_model: EmbeddingsModel, target_converter: TargetConverter, eos_token: int = 8,
                 padding_value: int = -100) -> None:
        super().__init__(embeddings_model, target_converter, eos_token)

        self.padding_value = padding_value

    def __call__(self, pairs_batch):
        """
        Method to call the vectorizer over a pairs of batch elements where the first one is the address and the second
        is a list of target tags.
        Args:
            pairs_batch (): The paired elements to vectorize.

        Return:
            The sorted element converter into either embeddings vector or target idx.
        """
        batch = []
        self._max_length = 0

        for pair in pairs_batch:
            input_sequence, word_decomposition_lengths = self._vectorize_sequence(pair[0])

            target_sequence = self._convert_target(pair[1])

            batch.append((input_sequence, word_decomposition_lengths, target_sequence))

        # todo in a method
        for decomposed_sequence, _, _ in batch:
            for decomposition in decomposed_sequence:
                if len(decomposition) != self._max_length:
                    for i in range(self._max_length - len(decomposition)):
                        decomposition.append(
                            np.ones(self.embeddings_model.dim) * self.padding_value)  # todo validate if the dim is ok

        return sorted(batch, key=lambda x: len(x[0]), reverse=True)

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
