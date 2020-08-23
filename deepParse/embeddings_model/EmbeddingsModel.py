from abc import ABC, abstractmethod


class EmbeddingsModel(ABC):
    """
    Abstract class for callable embeddings model.
    """

    @abstractmethod
    def __call__(self, pairs_batch):
        pass
