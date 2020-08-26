from abc import ABC, abstractmethod
from typing import List

from ..embeddings_model.embeddings_model import EmbeddingsModel


class Vectorizer(ABC):
    """
    Vectorizer abstract class to vectorize an address into a list of embeddings.

    Args:
        embeddings_model (~deepParse.embeddings_model.EmbeddingsModel): A callable embeddings model.

    """

    def __init__(self, embeddings_model: EmbeddingsModel) -> None:
        self.embeddings_model = embeddings_model
        self.eos_token = 8

    @abstractmethod
    def __call__(self, addresses: List[str]) -> List:
        """
        Method to vectorizer addresses.

        Args:
            addresses (List[str]): The addresses to vectorize.

        Return:
            The addresses elements (components) embeddings vector.
        """

        pass
