from typing import List

from pymagnitudelight import Magnitude

from deepparse.embeddings_models import EmbeddingsModel, ndarray


class MagnitudeEmbeddingsModel(EmbeddingsModel):
    """
        FastText embeddings network from `Enriching Word Vectors with Subword Information
        <https://arxiv.org/abs/1607.04606>`_ using the magnitude mapping
        (<https://github.com/plasticityai/magnitude>`_), which reduce memory footprint.
    """

    def __init__(self, embeddings_path: str) -> None:
        super().__init__()
        self.model = Magnitude(embeddings_path)

        self.model.dim = 300  # fastText is only in 300d

    def __call__(self, words: str) -> List[ndarray]:
        """
        Callable method to get word vector of a complete address.

        Args:
            words (str): Address to get vector for words.

        Return:
            The fastText embedding for a list of words.
        """
        # we leverage the multiple word query which are faster than single word query
        return self.model.query(words.split())
