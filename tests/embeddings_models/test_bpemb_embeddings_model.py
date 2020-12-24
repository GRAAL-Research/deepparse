import unittest
from unittest import TestCase
from unittest.mock import patch, Mock

from deepparse.embeddings_models import BPEmbEmbeddingsModel


class BPEmbEmbeddingsModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_word = "test"

    def setUp(self):
        self.model = Mock()
        self.model.dim = 9

    def test_whenInstantiatedWithPath_thenShouldLoadBPEmbModel(self):
        with patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb", return_value=self.model) as loader:
            self.embeddings_model = BPEmbEmbeddingsModel(verbose=False, lang="multi", vs=100000, dim=300)

            loader.assert_called_with(lang="multi", vs=100000, dim=300)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb", return_value=self.model):
            self.embeddings_model = BPEmbEmbeddingsModel(verbose=False, lang="multi", vs=100000, dim=300)

            self.embeddings_model(self.a_word)

            self.model.embed.assert_called_with(self.a_word)


if __name__ == "__main__":
    unittest.main()
