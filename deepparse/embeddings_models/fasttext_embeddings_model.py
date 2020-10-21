from numpy.core.multiarray import ndarray

from .embeddings_model import EmbeddingsModel
from .magnitude_wrapper import MagnitudeWrapper
from .. import load_fasttext_embeddings


class FastTextEmbeddingsModel(EmbeddingsModel):
    """
    FastText embeddings network from `Enriching Word Vectors with Subword Information
    <https://arxiv.org/abs/1607.04606>`_.

    Args:
       embeddings_path (str): Path to the bin embeddings vector (.bin).
       magnitude (bool): Either or not to use a magnitude mapping (<https://github.com/plasticityai/magnitude>`_),
            which reduce memory footprint.
    """

    def __init__(self, embeddings_path: str, magnitude: bool) -> None:
        super().__init__()
        if magnitude:
            self.model = MagnitudeWrapper(embeddings_path)
        else:
            self.model = load_fasttext_embeddings(embeddings_path)

        self.model.dim = 300  # fastText is only in 300d

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The fastText embedding for a word.
        """
        return self.model[word]
