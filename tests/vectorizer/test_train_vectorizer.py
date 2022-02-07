from unittest import TestCase
from unittest.mock import MagicMock

from deepparse.data_error import DataError
from deepparse.embeddings_models import EmbeddingsModel
from deepparse.vectorizer import TrainVectorizer, BPEmbVectorizer


class TrainVectorizerTest(TestCase):
    def test_givenAEmbeddingVectorizer_whenCallVectorizer_thenProcess(self):
        train_vectorizer = TrainVectorizer(MagicMock(), MagicMock())

        output = train_vectorizer(["A list"])

        self.assertIsInstance(output, zip)

    def test_givenAVectorizer_whenCallAnAddress_thenProcess(self):
        train_vectorizer = TrainVectorizer(MagicMock(side_effect=[[0]]), MagicMock(side_effect=[0, 0]))

        output = train_vectorizer(["A list of"])
        self.assertEqual(list(output), [(0, [0, 0])])

    def test_givenAVectorizer_whenCallWithAnWhiteSpaceOnlyAddress_thenRaiseError(self):
        embedding_network = MagicMock(spec=EmbeddingsModel)
        embedding_network.dim = 2
        bpemb_vectorizer = BPEmbVectorizer(embedding_network)

        train_vectorizer = TrainVectorizer(bpemb_vectorizer, MagicMock())
        a_whitespace_only_address = " "
        with self.assertRaises(DataError):
            train_vectorizer([a_whitespace_only_address])
