import platform

from gensim.models.fasttext import load_facebook_vectors
from numpy.core.multiarray import ndarray

from .embeddings_model import EmbeddingsModel
from .. import load_fasttext_embeddings


class FastTextEmbeddingsModel(EmbeddingsModel):
    """
    FastText embeddings network from `Enriching Word Vectors with Subword Information
    <https://arxiv.org/abs/1607.04606>`_.

    Args:
       embeddings_path (str): Path to the bin embeddings vector (.bin).
       verbose (bool): Either or not to make the loading of the embeddings verbose.

    Note:
        Since Windows uses `spawn` instead of `fork` during multiprocess (for the data loading pre-processing
        `num_worker` > 0) we use the Gensim model, which takes more RAM (~10 GO) than the Fasttext one (~8 GO).
        It also takes a longer time to load. See here the
        `issue <https://github.com/GRAAL-Research/deepparse/issues/89>`_.
    """

    def __init__(self, embeddings_path: str, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)

        if platform.system() == "Windows":
            self.model = load_facebook_vectors(embeddings_path)
        else:
            self.model = load_fasttext_embeddings(embeddings_path)

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The fastText embedding for a word.
        """
        return self.model[word]

    @property
    def dim(self):
        return self.model.get_dimension()
