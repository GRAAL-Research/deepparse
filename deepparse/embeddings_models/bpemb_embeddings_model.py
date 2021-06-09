import warnings

from bpemb import BPEmb
from numpy.core.multiarray import ndarray

from .embeddings_model import EmbeddingsModel


class BPEmbEmbeddingsModel(EmbeddingsModel):
    """
    BPEmb embeddings network from `BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages
    <https://www.aclweb.org/anthology/L18-1473/>`_. The arguments are the same as the
    `BPEmb class <https://github.com/bheinzerling/bpemb/blob/master/bpemb/bpemb.py>`_

    Params:
        verbose (bool): Either or not to make the loading of the embeddings verbose.
    """

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)
        with warnings.catch_warnings():
            # annoying scipy.sparcetools private module warnings removal
            # annoying boto warnings
            warnings.filterwarnings("ignore")
            model = BPEmb(lang="multi", vs=100000, dim=300)  # defaults parameters
        self.model = model

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The BP embedding for a word.
        """
        return self.model.embed(word)
