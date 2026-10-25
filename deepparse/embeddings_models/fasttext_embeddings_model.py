from numpy import ndarray

from .embeddings_model import EmbeddingsModel
from .fasttext_bin_reader import FastTextBinReader


class FastTextEmbeddingsModel(EmbeddingsModel):
    """
    FastText embeddings network from `Enriching Word Vectors with Subword Information
    <https://arxiv.org/abs/1607.04606>`_.

    Args:
       embeddings_path (str): Path to the bin embeddings vector (.bin).
       verbose (bool): Either or not to make the loading of the embeddings verbose.

    Note:
        The embeddings are read with our own reader of the fastText ``.bin`` format
        (:class:`~deepparse.embeddings_models.FastTextBinReader`), which gives the same vectors as the ``fasttext``
        package without depending on it nor on ``gensim``. The input matrix is memory-mapped, so the loading is fast
        and the data loading workers share its memory.
    """

    def __init__(self, embeddings_path: str, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)

        self.model = FastTextBinReader(embeddings_path)

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The FastText embedding for a word.
        """
        return self.model[word]

    @property
    def dim(self) -> int:
        return self.model.get_dimension()
