import unittest
from unittest import TestCase
from unittest.mock import patch, Mock

from deepparse.embeddings_models import BPEmbEmbeddingsModel


class BPEmbEmbeddingsModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_word = "test"
        cls.dim = 9

    def setUp(self):
        self.model = Mock()
        self.model.dim = self.dim

    def test_whenInstantiatedWithPath_thenShouldLoadBPEmbModel(self):
        with patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb", return_value=self.model) as loader:
            _ = BPEmbEmbeddingsModel(verbose=False)

            loader.assert_called_with(lang="multi", vs=100000, dim=300)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb", return_value=self.model):
            embeddings_model = BPEmbEmbeddingsModel(verbose=False)

            embeddings_model(self.a_word)

            self.model.embed.assert_called_with(self.a_word)

    def test_givenADimOf9_whenAskDimProperty_thenReturnProperDim(self):
        with patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb", return_value=self.model):
            embeddings_model = BPEmbEmbeddingsModel(verbose=False)

            actual = embeddings_model.dim
            expected = self.dim
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
