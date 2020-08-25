from abc import ABC, abstractmethod

from numpy.core.multiarray import ndarray


class EmbeddingsModel(ABC):
    """
    Abstract (wrapper) class for callable embeddings model.
    """

    @abstractmethod
    def __call__(self, word: str) -> ndarray:
        pass
