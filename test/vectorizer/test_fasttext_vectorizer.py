import unittest
from unittest import TestCase
from unittest.mock import Mock

from deepparse.vectorizer import FastTextVectorizer
from deepparse.embeddings_models import EmbeddingsModel


class FasttextVectorizerTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A_EMBEDDING_MATRIX = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 1], [2, 2], [1, 2], [2, 1]]
        cls.A_ADDRESS = ['5 test street']
        cls.A_VECTORIZED_ADDRESS = [[[0, 0], [0, 1], [1, 0]]]
        cls.A_ADDRESS_LIST = ['3 test way', '2 test road quebec']
        cls.A_VECTORIZED_ADDRESS_LIST = [[[0, 0], [0, 1], [1, 0]], [[1, 1], [0, 2], [2, 1], [2, 2]]]

    def setUp(self):
        self.embedding_network = Mock(spec=EmbeddingsModel, side_effect=self.A_EMBEDDING_MATRIX)
        self.fasttext_vectorizer = FastTextVectorizer(self.embedding_network)

    def test_givenAnAddress_whenVectorizingTheAddress_thenShouldCallEmbeddingModelForEachWord(self):
        self.fasttext_vectorizer(self.A_ADDRESS)

        self.assertEqual(self.embedding_network.call_count, len(self.A_ADDRESS[0].split()))

    def test_givenAnAddress_whenVectorizingTheAddress_thenShouldReturnCorrectEmbeddings(self):
        embedded_address = self.fasttext_vectorizer(self.A_ADDRESS)

        self.assertEqual(embedded_address, self.A_VECTORIZED_ADDRESS)

    def test_givenAnAddressList_whenVectorizingTheAddress_thenShouldReturnCorrectEmbeddings(self):
        embedded_address = self.fasttext_vectorizer(self.A_ADDRESS_LIST)

        self.assertEqual(embedded_address, self.A_VECTORIZED_ADDRESS_LIST)


if __name__ == '__main__':
    unittest.main()