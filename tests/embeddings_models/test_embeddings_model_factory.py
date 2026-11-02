# pylint: disable=line-too-long, unused-argument
import unittest
from unittest import TestCase
from unittest.mock import patch

from deepparse.embeddings_models import (
    BPEmbEmbeddingsModel,
    EmbeddingsModelFactory,
    FastTextEmbeddingsModel,
    MagnitudeEmbeddingsModel,
    Model2VecEmbeddingsModel,
)


class EmbeddingsModelFactoryTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_bpemb_embeddings_model_type = "bpemb"
        cls.a_fasttext_embeddings_model_type = "fasttext"
        cls.a_fasttext_magnitude_embeddings_model_type = "fasttext-light"

        cls.an_invalid_embeddings_model_type = "invalid"

        cls.a_cache_dir = "~/.cache/deepparse"

    def setUp(self):
        self.embeddings_model_factory = EmbeddingsModelFactory()

    @patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmbLoader")
    def test_givenABpembEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldReturnCorrectEmbeddingsModel(
        self, bpemb_mock
    ):
        embeddings_model = self.embeddings_model_factory.create(self.a_bpemb_embeddings_model_type, self.a_cache_dir)

        self.assertIsInstance(embeddings_model, BPEmbEmbeddingsModel)

    @patch("deepparse.embeddings_models.embeddings_model_factory.download_fasttext_embeddings")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.FastTextBinReader")
    def test_givenAFasttextEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldReturnCorrectEmbeddingsModel(
        self, reader_mock, download_mock
    ):
        embeddings_model = self.embeddings_model_factory.create(self.a_fasttext_embeddings_model_type, self.a_cache_dir)

        self.assertIsInstance(embeddings_model, FastTextEmbeddingsModel)

    @patch("deepparse.embeddings_models.embeddings_model_factory.download_fasttext_magnitude_embeddings")
    @patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude")
    def test_givenAFasttextMagnitudeEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldReturnCorrectEmbeddingsModel(
        self, download_mock, load_mock
    ):
        embeddings_model = self.embeddings_model_factory.create(
            self.a_fasttext_magnitude_embeddings_model_type, self.a_cache_dir
        )

        self.assertIsInstance(embeddings_model, MagnitudeEmbeddingsModel)

    @patch(
        "deepparse.embeddings_models.embeddings_model_factory.download_model2vec_embeddings",
        return_value=("a/path", 256),
    )
    @patch("model2vec.StaticModel.from_pretrained")
    def test_givenAModel2VecEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldReturnCorrectEmbeddingsModel(
        self, load_mock, download_mock
    ):
        embeddings_model = self.embeddings_model_factory.create("model2vec-8m_attention", self.a_cache_dir)

        self.assertIsInstance(embeddings_model, Model2VecEmbeddingsModel)
        download_mock.assert_called_once_with("model2vec-8m_attention", cache_dir=self.a_cache_dir, verbose=True)
        load_mock.assert_called_once_with("a/path")

    def test_givenAnInvalidEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.embeddings_model_factory.create(self.an_invalid_embeddings_model_type, self.a_cache_dir)


if __name__ == "__main__":
    unittest.main()
