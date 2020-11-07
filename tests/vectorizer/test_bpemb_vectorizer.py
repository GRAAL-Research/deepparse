import unittest
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from deepparse.embeddings_models.embeddings_model import EmbeddingsModel
from deepparse.vectorizer import BPEmbVectorizer


class BpembVectorizerTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_embedding_matrix = [[np.array([0, 0]), np.array([1, 1])],
                                  [np.array([0, 1]), np.array([1, 2]),
                                   np.array([2, 3])], [np.array([1, 0])], [np.array([1, 1]),
                                                                           np.array([2, 2])], [np.array([0, 2])],
                                  [np.array([2, 1])], [np.array([2, 2]), np.array([3, 3])], [np.array([1, 2])],
                                  [np.array([2, 1])]]
        cls.a_address = ["5 test street"]
        cls.a_vectorized_address = [[np.array([0, 0]), np.array([1, 1]),
                                     np.zeros(2)], [np.array([0, 1]),
                                                    np.array([1, 2]),
                                                    np.array([2, 3])], [[1, 0], np.zeros(2),
                                                                        np.zeros(2)]]
        cls.a_address_list = ["3 test way", "2 test road quebec"]
        cls.a_vectorized_address_list = [[[np.array([0, 0]), np.array([1, 1]),
                                           np.zeros(2)], [np.array([0, 1]),
                                                          np.array([1, 2]),
                                                          np.array([2, 3])], [[1, 0], np.zeros(2),
                                                                              np.zeros(2)]],
                                         [[np.array([1, 1]), np.array([2, 2]),
                                           np.zeros(2)], [np.array([0, 2]), np.zeros(2),
                                                          np.zeros(2)], [np.array([2, 1]),
                                                                         np.zeros(2),
                                                                         np.zeros(2)],
                                          [np.array([2, 2]), np.array([3, 3]),
                                           np.zeros(2)]]]

    def setUp(self):
        self.embedding_network = Mock(spec=EmbeddingsModel, side_effect=self.a_embedding_matrix)
        self.embedding_network.dim = 2
        self.bpemb_vectorizer = BPEmbVectorizer(self.embedding_network)

    def test_givenAnAddress_whenVectorizingTheAddress_thenShouldCallEmbeddingModelForEachWord(self):
        self.bpemb_vectorizer(self.a_address)

        self.assertEqual(self.embedding_network.call_count, len(self.a_address[0].split()))

    def test_givenAnAddress_whenVectorizingTheAddress_thenShouldReturnCorrectEmbeddings(self):
        embedded_address = self.bpemb_vectorizer(self.a_address)
        embedded_address = embedded_address[0][0]

        self.assertTrue(self._compare_two_vectorized_addresses(embedded_address, self.a_vectorized_address))

    def test_givenAnAddressList_whenVectorizingTheAddress_thenShouldReturnCorrectEmbeddings(self):
        embedded_address = self.bpemb_vectorizer(self.a_address_list)

        self.assertTrue(
            self._compare_two_vectorized_addresses(embedded_address[0][0], self.a_vectorized_address_list[0]))
        self.assertTrue(
            self._compare_two_vectorized_addresses(embedded_address[1][0], self.a_vectorized_address_list[1]))

    def _compare_two_vectorized_addresses(self, embedded_address, vectorized_address):
        res = True
        for embedded_word, word_truth in zip(embedded_address, vectorized_address):
            for subword_embedding, subword_truth in zip(embedded_word, word_truth):
                res = np.array_equal(subword_embedding, subword_truth)

        return res


if __name__ == "__main__":
    unittest.main()
