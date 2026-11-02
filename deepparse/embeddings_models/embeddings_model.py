import logging
from abc import ABC, abstractmethod

from numpy import ndarray

logger = logging.getLogger(__name__)


class EmbeddingsModel(ABC):
    """
    Abstract (wrapper) class for callable embeddings network.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.model = None
        if verbose:
            logger.info("Loading the embeddings model")

    @abstractmethod
    def __call__(self, words: str) -> ndarray:
        pass

    @property
    def dim(self) -> int:
        return self.model.dim
