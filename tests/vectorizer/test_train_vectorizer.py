from unittest import TestCase
from unittest.mock import MagicMock

from deepparse.vectorizer import TrainVectorizer


class TrainVectorizerTest(TestCase):

    def test_givenAEmbeddingVectorizer_whenCallVectorizer_thenProcess(self):
        train_vectorizer = TrainVectorizer(MagicMock(), MagicMock())

        output = train_vectorizer(["A list"])

        self.assertIsInstance(output, zip)

    def test_givenAVectorizer_whenCallAnAddress_thenProcess(self):
        train_vectorizer = TrainVectorizer(MagicMock(side_effect=[[0]]), MagicMock(side_effect=[0, 0]))

        output = train_vectorizer(["A list of"])
        self.assertEqual(list(output), [(0, [0, 0])])
