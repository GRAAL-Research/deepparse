# pylint: disable=line-too-long, unused-argument
import unittest
from unittest import TestCase
from unittest.mock import patch

from deepparse.embeddings_models import (
    EmbeddingsModelFactory,
    FastTextEmbeddingsModel,
    BPEmbEmbeddingsModel,
    MagnitudeEmbeddingsModel,
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

    @patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb")
    def test_givenABpembEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldReturnCorrectEmbeddingsModel(
        self, bpemb_mock
    ):
        embeddings_model = self.embeddings_model_factory.create(self.a_bpemb_embeddings_model_type, self.a_cache_dir)

        self.assertIsInstance(embeddings_model, BPEmbEmbeddingsModel)

    @patch("deepparse.embeddings_models.embeddings_model_factory.download_fasttext_embeddings")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.load_facebook_vectors")
    def test_givenAFasttextEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldReturnCorrectEmbeddingsModel(
        self, facebook_vectors_load_mock, fasttext_load_mock, download_mock
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

    def test_givenAnInvalidEmbeddingsModelType_whenCreatingEmbeddingsModel_thenShouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.embeddings_model_factory.create(self.an_invalid_embeddings_model_type, self.a_cache_dir)


if __name__ == "__main__":
    unittest.main()
