from abc import ABC, abstractmethod
from typing import List, Union

from numpy.core.multiarray import ndarray


class EmbeddingsModel(ABC):
    """
    Abstract (wrapper) class for callable embeddings network.
    """

    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def __call__(self, word: Union[str, List[str]]) -> Union[ndarray, List[ndarray]]:
        pass

    @property
    def dim(self):
        return self.model.dim
