import unittest
from unittest import TestCase
from unittest.mock import Mock

from deepparse.embeddings_models import EmbeddingsModel
from deepparse.vectorizer import MagnitudeVectorizer


class MagnitudeVectorizerTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_embedding_matrix = [[[0, 0], [0, 1], [1, 0]], [[1, 1], [0, 2], [2, 1], [2, 2]]]
        cls.a_address = ["5 test street"]
        cls.a_vectorized_address = [[[0, 0], [0, 1], [1, 0]]]
        cls.a_list_address = ["3 test way", "2 test road quebec"]
        cls.a_vectorized_address_list = [[[0, 0], [0, 1], [1, 0]], [[1, 1], [0, 2], [2, 1], [2, 2]]]

    def setUp(self):
        self.embedding_network = Mock(spec=EmbeddingsModel, side_effect=self.a_embedding_matrix)
        self.magnitude_vectorizer = MagnitudeVectorizer(self.embedding_network)

    def test_givenAnAddress_whenVectorizingTheAddress_thenShouldCallEmbeddingModelInBatch(self):
        self.magnitude_vectorizer(self.a_address)

        self.assertEqual(self.embedding_network.call_count, len(self.a_address))  # since in batch

    def test_givenAnAddress_whenVectorizingTheAddress_thenShouldReturnCorrectEmbeddings(self):
        embedded_address = self.magnitude_vectorizer(self.a_address)

        self.assertEqual(embedded_address, self.a_vectorized_address)

    def test_givenAnAddressList_whenVectorizingTheAddress_thenShouldReturnCorrectEmbeddings(self):
        embedded_address = self.magnitude_vectorizer(self.a_list_address)

        self.assertEqual(embedded_address, self.a_vectorized_address_list)


if __name__ == "__main__":
    unittest.main()
