import fasttext
import fasttext.util
from numpy.core.multiarray import ndarray

from deepParse.embeddings_model import EmbeddingsModel


class FastTextEmbeddingsModel(EmbeddingsModel):
    """
    FastText embeddings model from `Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_.

    Args:
       embeddings_path (str): Path to the bin embeddings vector (.bin).
    """

    def __init__(self, embeddings_path: str) -> None:
        self.model = fasttext.load_model(embeddings_path)

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The fastText embedding for a word.
        """
        return self.model[word]  # verify output format
