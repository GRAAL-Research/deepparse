import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from deepparse.embeddings_models import FastTextEmbeddingsModel


class FasttextEmbeddingsModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_path = "."
        cls.a_word = "test"
        cls.verbose = False
        cls.dim = 9

    def setUp(self):
        model = MagicMock()
        model.get_dimension.return_value = self.dim
        self.model = model

    def test_whenInstantiatedWithPath_thenShouldLoadTheEmbeddingsWithTheBinReader(self):
        with patch(
            "deepparse.embeddings_models.fasttext_embeddings_model.FastTextBinReader",
            return_value=self.model,
        ) as reader:
            FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            reader.assert_called_once_with(self.a_path)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch(
            "deepparse.embeddings_models.fasttext_embeddings_model.FastTextBinReader",
            return_value=self.model,
        ):
            embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            embeddings_model(self.a_word)

            self.model.__getitem__.assert_called_with(self.a_word)

    def test_givenADimOf9_whenAskDimProperty_thenReturnProperDim(self):
        with patch(
            "deepparse.embeddings_models.fasttext_embeddings_model.FastTextBinReader",
            return_value=self.model,
        ):
            embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            self.assertEqual(self.dim, embeddings_model.dim)


if __name__ == "__main__":
    unittest.main()
