import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

from deepparse.embeddings_models import FastTextEmbeddingsModel


class FasttextEmbeddingsModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_path = "."
        cls.a_word = "test"

    def setUp(self):
        self.model = MagicMock()
        self.model.dim = 9

    def test_whenInstanciatedWithPath_thenShouldLoadFasttextModel(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model) as loader:
            self.embeddings_model = FastTextEmbeddingsModel(self.a_path)

            loader.assert_called_with(self.a_path)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model):
            self.embeddings_model = FastTextEmbeddingsModel(self.a_path)

            self.embeddings_model(self.a_word)

            self.model.__getitem__.assert_called_with(self.a_word)


if __name__ == "__main__":
    unittest.main()
