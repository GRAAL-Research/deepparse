import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from deepparse.embeddings_models import Model2VecEmbeddingsModel


class Model2VecEmbeddingsModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_path = "a/local/model2vec/snapshot"
        cls.a_word = "test"
        cls.dim = 256

    def setUp(self):
        self.static_model = MagicMock()
        self.static_model.dim = self.dim
        self.static_model.encode.return_value = np.ones((2, self.dim))

    def test_whenInstantiatedWithPath_thenShouldLoadTheModelFromThePath(self):
        with patch("model2vec.StaticModel.from_pretrained", return_value=self.static_model) as loader:
            Model2VecEmbeddingsModel(self.a_path, verbose=False)

            loader.assert_called_once_with(self.a_path)

    def test_whenEncodingWords_thenShouldEncodeWithoutModel2VecMultiprocessing(self):
        with patch("model2vec.StaticModel.from_pretrained", return_value=self.static_model):
            embeddings_model = Model2VecEmbeddingsModel(self.a_path, verbose=False)

            embeddings_model.encode(["a", "b"])

            self.static_model.encode.assert_called_once_with(["a", "b"], use_multiprocessing=False)

    def test_whenCalledToEmbedAWord_thenShouldReturnTheWordVector(self):
        with patch("model2vec.StaticModel.from_pretrained", return_value=self.static_model):
            embeddings_model = Model2VecEmbeddingsModel(self.a_path, verbose=False)

            word_vector = embeddings_model(self.a_word)

            self.static_model.encode.assert_called_once_with([self.a_word], use_multiprocessing=False)
            self.assertEqual((self.dim,), word_vector.shape)

    def test_whenGettingTheDimension_thenShouldReturnTheModelDimension(self):
        with patch("model2vec.StaticModel.from_pretrained", return_value=self.static_model):
            embeddings_model = Model2VecEmbeddingsModel(self.a_path, verbose=False)

            self.assertEqual(self.dim, embeddings_model.dim)

    def test_givenModel2VecNotInstalled_whenInstantiated_thenRaiseAnImportErrorWithTheInstallCommand(self):
        with patch.dict("sys.modules", {"model2vec": None}):
            with self.assertRaisesRegex(ImportError, r"deepparse\[model2vec\]"):
                Model2VecEmbeddingsModel(self.a_path, verbose=False)


if __name__ == "__main__":
    unittest.main()
