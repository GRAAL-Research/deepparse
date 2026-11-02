# pylint: disable=unused-argument, arguments-differ
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from deepparse.embeddings_models import (
    BPEmbEmbeddingsModel,
    FastTextEmbeddingsModel,
    MagnitudeEmbeddingsModel,
    Model2VecEmbeddingsModel,
)
from deepparse.vectorizer import (
    BPEmbVectorizer,
    FastTextVectorizer,
    MagnitudeVectorizer,
    Model2VecVectorizer,
    VectorizerFactory,
)


class VectorizerFactoryTest(TestCase):
    @classmethod
    @patch("deepparse.embeddings_models.bpemb_embeddings_model.BPEmbLoader")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.FastTextBinReader")
    @patch("deepparse.embeddings_models.magnitude_embeddings_model.Magnitude")
    def setUpClass(cls, magnitude_mock, reader_mock, bpemb_mock):
        a_cache_dir = "~/.cache/deepparse"
        cls.a_bpemb_embeddings_model = BPEmbEmbeddingsModel(a_cache_dir)

        a_embeddings_path = "path"
        cls.a_fasttext_embeddings_model = FastTextEmbeddingsModel(a_embeddings_path)

        cls.a_magnitude_embeddings_model = MagnitudeEmbeddingsModel(a_embeddings_path)

        cls.an_unsupported_embeddings_model = "unsupported"

    def setUp(self):
        self.vectorizer_factory = VectorizerFactory()

    def test_givenABpembEmbeddingsModel_whenCreatingVectorizer_thenShouldReturnProperVectorizer(self):
        vectorizer = self.vectorizer_factory.create(self.a_bpemb_embeddings_model)

        self.assertIsInstance(vectorizer, BPEmbVectorizer)

    def test_givenAFasttextEmbeddingsModel_whenCreatingVectorizer_thenShouldReturnProperVectorizer(self):
        vectorizer = self.vectorizer_factory.create(self.a_fasttext_embeddings_model)

        self.assertIsInstance(vectorizer, FastTextVectorizer)

    def test_givenAMagnitudeEmbeddingsModel_whenCreatingVectorizer_thenShouldReturnProperVectorizer(self):
        vectorizer = self.vectorizer_factory.create(self.a_magnitude_embeddings_model)

        self.assertIsInstance(vectorizer, MagnitudeVectorizer)

    def test_givenAModel2VecEmbeddingsModel_whenCreatingVectorizer_thenShouldReturnProperVectorizer(self):
        vectorizer = self.vectorizer_factory.create(MagicMock(spec=Model2VecEmbeddingsModel))

        self.assertIsInstance(vectorizer, Model2VecVectorizer)

    def test_givenAUnsupportedEmbeddingsModel_whenCreatingVectorizer_thenShouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.vectorizer_factory.create(self.an_unsupported_embeddings_model)


if __name__ == "__main__":
    unittest.main()
