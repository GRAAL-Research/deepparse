from typing import List

from .vectorizer import Vectorizer
from .. import validate_data_to_parse


class FastTextVectorizer(Vectorizer):
    """
    FastText vectorizer to convert an address into fastText embeddings.
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

        return [self._vectorize_sequence(address) for address in addresses]

    def _vectorize_sequence(self, address: str) -> List:
        """
        Method to vectorize the address.

        Args:
            address (str): Address to vectorize using fastText.

        Return:
            A list of word vector.
        """
        return [self.embeddings_model(word) for word in address.split()]
