import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from deepparse.vectorizer import VectorizerFactory, BPEmbVectorizer, MagnitudeVectorizer, FastTextVectorizer
from deepparse.embeddings_models import (
    BPEmbEmbeddingsModel,
    FastTextEmbeddingsModel,
    MagnitudeEmbeddingsModel,
)


class VectorizerFactoryTest(TestCase):
    @classmethod
    @patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmb")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings")
    @patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude")
    def setUpClass(cls, bpemb_mock, fasttext_load_mock, magnitude_mock):
        cls.a_bpemb_embeddings_model_type = "bpemb"
        cls.a_fasttext_embeddings_model_type = "fasttext"
        cls.a_fasttext_magnitude_embeddings_model_type = "fasttext_magnitude"

        cls.a_cache_dir = "~/.cache/deepparse"

        cls.a_verbose = True

        cls.a_bpemb_embeddings_model = BPEmbEmbeddingsModel(cls.a_cache_dir)

        a_embeddings_path = "path"
        cls.a_fasttext_embedding_model = FastTextEmbeddingsModel(a_embeddings_path)

        cls.a_magnitude_embeddings_model = MagnitudeEmbeddingsModel(a_embeddings_path)

        cls.an_unsupported_embeddings_model_type = "a_model_type"
        cls.an_unsupported_embeddings_model = "a_model"

    def setUp(self):
        self.embedding_factory_mock = Mock()

        self.embedding_factory_mock.create.side_effect = lambda *params: {
            (self.a_bpemb_embeddings_model_type, self.a_cache_dir, self.a_verbose): self.a_bpemb_embeddings_model,
            (self.a_fasttext_embeddings_model_type, self.a_cache_dir, self.a_verbose): self.a_fasttext_embedding_model,
            (
                self.a_fasttext_magnitude_embeddings_model_type,
                self.a_cache_dir,
                self.a_verbose,
            ): self.a_magnitude_embeddings_model,
            (
                self.an_unsupported_embeddings_model_type,
                self.a_cache_dir,
                self.a_verbose,
            ): self.an_unsupported_embeddings_model,
        }[params]

        self.vectorizer_factory = VectorizerFactory(self.embedding_factory_mock)

    def test_givenABpembEmbeddingsModelType_whenCreatingVectorizer_thenShouldReturnPropoerVectorizer(self):
        vectorizer = self.vectorizer_factory.create(
            self.a_bpemb_embeddings_model_type, self.a_cache_dir, self.a_verbose
        )

        self.assertIsInstance(vectorizer, BPEmbVectorizer)

    def test_givenAFasttextEmbeddingsModelType_whenCreatingVectorizer_thenShouldReturnPropoerVectorizer(self):
        vectorizer = self.vectorizer_factory.create(
            self.a_fasttext_embeddings_model_type, self.a_cache_dir, self.a_verbose
        )

        self.assertIsInstance(vectorizer, FastTextVectorizer)

    def test_givenAMagnitudeEmbeddingsModelType_whenCreatingVectorizer_thenShouldReturnPropoerVectorizer(self):
        vectorizer = self.vectorizer_factory.create(
            self.a_fasttext_magnitude_embeddings_model_type, self.a_cache_dir, self.a_verbose
        )

        self.assertIsInstance(vectorizer, MagnitudeVectorizer)

    def test_givenAUnsupportedEmbeddingsModelType_whenCreatingVectorizer_thenShouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.vectorizer_factory.create(self.an_unsupported_embeddings_model_type, self.a_cache_dir, self.a_verbose)


if __name__ == "__main__":
    unittest.main()
