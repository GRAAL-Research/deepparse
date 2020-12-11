from abc import ABC, abstractmethod

from numpy.core.multiarray import ndarray


class EmbeddingsModel(ABC):
    """
    Abstract (wrapper) class for callable embeddings network.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.model = None
        if verbose:
            print("Loading the embeddings model")

    @abstractmethod
    def __call__(self, words: str) -> ndarray:
        pass

    @property
    def dim(self):
        return self.model.dim
