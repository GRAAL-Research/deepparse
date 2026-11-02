import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from deepparse.embeddings_models import Model2VecEmbeddingsModel
from deepparse.errors.data_error import DataError
from deepparse.vectorizer import Model2VecVectorizer


class Model2VecVectorizerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_address_list = ["3 test way", "2 test road quebec"]
        cls.a_words_matrix = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 1], [2, 2]])

    def setUp(self):
        self.embeddings_model = MagicMock(spec=Model2VecEmbeddingsModel)
        self.embeddings_model.encode.return_value = self.a_words_matrix
        self.model2vec_vectorizer = Model2VecVectorizer(self.embeddings_model)

    def test_given_a_whitespace_only_address_when_call_then_raise_error(self):
        with self.assertRaises(DataError):
            self.model2vec_vectorizer([" "])

    def test_givenAnAddressList_whenVectorizing_thenEncodeAllTheWordsInASingleCall(self):
        self.model2vec_vectorizer(self.a_address_list)

        self.embeddings_model.encode.assert_called_once_with(["3", "test", "way", "2", "test", "road", "quebec"])

    def test_givenAnAddressList_whenVectorizing_thenSplitTheWordsVectorsBackPerAddress(self):
        vectorized_addresses = self.model2vec_vectorizer(self.a_address_list)

        self.assertEqual(2, len(vectorized_addresses))
        np.testing.assert_array_equal(np.array(vectorized_addresses[0]), self.a_words_matrix[:3])
        np.testing.assert_array_equal(np.array(vectorized_addresses[1]), self.a_words_matrix[3:])


if __name__ == "__main__":
    unittest.main()
