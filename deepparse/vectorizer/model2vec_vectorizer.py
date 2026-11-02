from typing import List

from .. import validate_data_to_parse
from .vectorizer import Vectorizer


class Model2VecVectorizer(Vectorizer):
    """
    Model2Vec vectorizer to convert an address into Model2Vec embeddings.
    """

    def __call__(self, addresses: List[str]) -> List:
        """
        Method to vectorizer addresses.

        Args:
            addresses (list[str]): The addresses to vectorize.

        Return:
            A list of embeddings corresponding to the addresses' elements.
        """
        validate_data_to_parse(addresses)

        addresses_words = [address.split() for address in addresses]

        # We encode all the words of the batch in a single call, then split the vectors back per address.
        words_vectors = self.embeddings_model.encode([word for words in addresses_words for word in words])

        vectorized_addresses = []
        start = 0
        for words in addresses_words:
            end = start + len(words)
            vectorized_addresses.append(list(words_vectors[start:end]))
            start = end

        return vectorized_addresses
