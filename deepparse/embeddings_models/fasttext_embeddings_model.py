import platform
import warnings

from gensim.models.fasttext import load_facebook_vectors
from numpy import ndarray

from ..download_tools import FASTTEXT_AVAILABLE
from .embeddings_model import EmbeddingsModel


class FastTextEmbeddingsModel(EmbeddingsModel):
    """
    FastText embeddings network from `Enriching Word Vectors with Subword Information
    <https://arxiv.org/abs/1607.04606>`_.

    Args:
       embeddings_path (str): Path to the bin embeddings vector (.bin).
       verbose (bool): Either or not to make the loading of the embeddings verbose.

    Note:
        Since Windows uses ``spawn`` instead of ``fork`` during multiprocess (for the data loading pre-processing
        ``num_worker`` > 0), we use the Gensim model, which takes more RAM (~10 GO) than the Fasttext one (~8 GO).
        It also takes a longer time to load. See here the
        `issue <https://github.com/GRAAL-Research/deepparse/issues/89>`_.

    Note:
        On Python 3.13+, the ``fasttext`` package is not available because the underlying C++ library
        has not been updated for newer Python versions. In this case, the Gensim model is used automatically
        as a fallback. This fallback uses more RAM (~10 GO vs ~8 GO) and takes longer to load.
        You can install ``fasttext-wheel`` manually with ``pip install fasttext-wheel`` if a compatible
        version becomes available.
    """

    def __init__(self, embeddings_path: str, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)

        self._use_gensim = False

        if platform.system() == "Windows" or not FASTTEXT_AVAILABLE:
            if not FASTTEXT_AVAILABLE and platform.system() != "Windows":
                warnings.warn(
                    "The 'fasttext' package is not installed. Using gensim as a fallback to load "
                    "FastText embeddings. This uses more RAM (~10 GO vs ~8 GO) and is slower to load. "
                    "Install 'fasttext-wheel' for better performance if your Python version supports it.",
                    category=UserWarning,
                )
            self.model = load_facebook_vectors(embeddings_path)
            self._use_gensim = True
        else:
            from .. import load_fasttext_embeddings  # pylint: disable=import-outside-toplevel

            self.model = load_fasttext_embeddings(embeddings_path)

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
        if self._use_gensim:
            return self.model.vector_size
        return self.model.get_dimension()
