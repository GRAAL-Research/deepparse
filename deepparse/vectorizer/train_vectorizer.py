from typing import List, Iterable

from deepparse.converter import TagsConverter
from deepparse.vectorizer import Vectorizer


class TrainVectorizer:
    def __init__(self, embedding_vectorizer: Vectorizer, tags_converter: TagsConverter) -> None:
        """
        Vectorizer use during training to convert an address into word embeddings and to provide the target.
        """
        self.embedding_vectorizer = embedding_vectorizer
        self.tags_converter = tags_converter

    def __call__(self, addresses: List[str]) -> Iterable:
        """
        Method to vectorizer addresses for training.

        Args:
            addresses (list[str]): The addresses to vectorize.

        Return:
            A tuple compose of embeddings word addresses' and the target idxs.
        """
        input_sequence = []
        target_sequence = []

        input_sequence.extend(
            self.embedding_vectorizer([address[0] for address in addresses])
        )  # Need to be pass in batch

        # Otherwise, the padding for byte-pair encoding will be broken
        for address in addresses:
            target_tmp = [self.tags_converter(target) for target in address[1]]
            target_tmp.append(self.tags_converter("EOS"))  # to append the End Of Sequence token
            target_sequence.append(target_tmp)
        return zip(input_sequence, target_sequence)
