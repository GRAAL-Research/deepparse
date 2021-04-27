import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

from deepparse.embeddings_models import MagnitudeEmbeddingsModel


class MagnitudeEmbeddingsModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_path = "."
        cls.a_sentence_of_word = "a test sentence"
        cls.verbose = False
        cls.dim = 9

    def setUp(self):
        self.model = MagicMock()
        self.model.dim = self.dim

    def test_whenInstanciatedWithPath_thenShouldLoadFasttextModel(self):
        with patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude",
                   return_value=self.model) as loader:
            _ = MagnitudeEmbeddingsModel(self.a_path, verbose=self.verbose)

            loader.assert_called_with(self.a_path)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude", return_value=self.model):
            embeddings_model = MagnitudeEmbeddingsModel(self.a_path, verbose=self.verbose)

            embeddings_model(self.a_sentence_of_word)

            self.model.query.assert_called_with(self.a_sentence_of_word.split())

    def test_givenADimOf9_whenAskDimProperty_thenReturnProperDim(self):
        with patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude", return_value=self.model):
            embeddings_model = MagnitudeEmbeddingsModel(self.a_path, verbose=self.verbose)

            actual = embeddings_model.dim
            expected = self.dim
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
