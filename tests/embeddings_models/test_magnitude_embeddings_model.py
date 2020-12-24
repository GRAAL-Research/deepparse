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

    def setUp(self):
        self.model = MagicMock()
        self.model.dim = 9

    def test_whenInstanciatedWithPath_thenShouldLoadFasttextModel(self):
        with patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude",
                   return_value=self.model) as loader:
            self.embeddings_model = MagnitudeEmbeddingsModel(self.a_path, verbose=self.verbose)

            loader.assert_called_with(self.a_path)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude", return_value=self.model):
            self.embeddings_model = MagnitudeEmbeddingsModel(self.a_path, verbose=self.verbose)

            self.embeddings_model(self.a_sentence_of_word)

            self.model.query.assert_called_with(self.a_sentence_of_word.split())


if __name__ == "__main__":
    unittest.main()
