import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

from deepparse.embeddings_models import FastTextEmbeddingsModel


class FasttextEmbeddingsModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A_PATH = '.'
        cls.A_WORD = 'test'

    def setUp(self):
        self.model = MagicMock()
        self.model.dim = 9

    def test_whenInstanciatedWithPath_thenShouldLoadFasttextModel(self):
        with patch('deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings',
                   return_value=self.model) as loader:
            self.embeddings_model = FastTextEmbeddingsModel(self.A_PATH)

            loader.assert_called_with(self.A_PATH)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch('deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings',
                   return_value=self.model) as loader:
            self.embeddings_model = FastTextEmbeddingsModel(self.A_PATH)

            self.embeddings_model(self.A_WORD)

            self.model.__getitem__.assert_called_with(self.A_WORD)


if __name__ == '__main__':
    unittest.main()