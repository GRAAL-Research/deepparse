from typing import List

from numpy import ndarray

from .embeddings_model import EmbeddingsModel


class Model2VecEmbeddingsModel(EmbeddingsModel):
    """
    Model2Vec static embeddings network from `Model2Vec: Distill a Small Fast Model from any Sentence Transformer
    <https://github.com/MinishLab/model2vec>`_. A word vector is the mean of its subword tokens vectors.

    Args:
        embeddings_path (str): Path to the local directory of the Model2Vec model (e.g. a Hugging Face snapshot).
        verbose (bool): Either or not to make the loading of the embeddings verbose.

    Note:
        Model2Vec is an optional dependency. Install it with ``pip install deepparse[model2vec]``.
    """

    def __init__(self, embeddings_path: str, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)

        try:
            from model2vec import StaticModel  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise ImportError(
                "The 'model2vec' package is required for the model2vec models. "
                "Install it with 'pip install deepparse[model2vec]'."
            ) from error

        # We load from a local path since StaticModel.from_pretrained forces a download otherwise.
        self.model = StaticModel.from_pretrained(embeddings_path)

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The Model2Vec embedding for a word.
        """
        return self.encode([word])[0]

    def encode(self, words: List[str]) -> ndarray:
        """
        Method to get the vectors of many words in a single call, which is much faster than one call per word.

        Args:
            words (list[str]): The words to get the vectors.

        Return:
            A matrix of shape (number of words, embeddings dimension).
        """
        # The data loader already uses workers, so we disable the Model2Vec multiprocessing.
        return self.model.encode(words, use_multiprocessing=False)

    @property
    def dim(self) -> int:
        return self.model.dim
