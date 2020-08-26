from typing import List

from deepParse.vectorizer import Vectorizer


class FastTextVectorizer(Vectorizer):
    """
    FastText vectorizer to convert an address into fastText embeddings.
    """

    def __call__(self, addresses: List[str]) -> List:
        """
        Method to vectorizer addresses.

        Args:
            addresses (List[str]): The addresses to vectorize.

        Return:
            The addresses elements (components) embeddings vector.
        """
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
