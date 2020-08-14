from abc import ABC, abstractmethod

import fasttext
import fasttext.util
from bpemb import BPEmb


class EmbeddingsModel(ABC):
    """
    Abstract class for callable embeddings model.
    """

    @abstractmethod
    def __call__(self, pairs_batch):
        pass


class FastTextEmbeddingsModel(EmbeddingsModel):
    """
    FastText embeddings model from `Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_

    Args:
       embeddings_path (str): Path to the bin embeddings vector (.bin).
    """

    def __init__(self, embeddings_path: str) -> None:
        self.model = fasttext.load_model(embeddings_path)

    def __call__(self, word: str):
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.
        """
        return self.model[word]  # verify output format


class BPEmbEmbeddingsModel(EmbeddingsModel):
    """
    BPEmb embeddings model from `BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages <https://www.aclweb.org/anthology/L18-1473/>`_.
    The arguments are the same as the `BPEmb class <https://github.com/bheinzerling/bpemb/blob/master/bpemb/bpemb.py>`_

    """

    def __init__(self, **kwargs):
        self.model = BPEmb(**kwargs)

    def __call__(self, word: str):
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.
        """
        return self.model.embed(word)

    @property
    def dim(self):
        return self.model.dim
