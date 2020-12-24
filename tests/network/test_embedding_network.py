import unittest
from unittest import TestCase
from unittest.mock import patch

from deepparse.network import EmbeddingNetwork


class EmbeddingNetworkTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_input_size = 2
        cls.a_hidden_size = 2
        cls.a_projection_size = 2
        cls.a_number_of_layers = 1
        cls.a_maxpool_kernel_size = 5

    def setUp(self):
        self.embedding_network = EmbeddingNetwork(self.a_input_size, self.a_hidden_size, self.a_projection_size)

    @patch('deepparse.network.embedding_network.torch.nn.LSTM')
    def test_whenInstanciatingEmbeddingNetwork_thenShouldInstanciateLSTMWithCorrectParameters(self, lstm_mock):
        self.embedding_network = EmbeddingNetwork(self.a_input_size, self.a_hidden_size, self.a_projection_size,
                                                  self.a_number_of_layers)

        lstm_mock.assert_called_with(self.a_input_size,
                                     self.a_hidden_size,
                                     num_layers=self.a_number_of_layers,
                                     batch_first=True,
                                     bidirectional=True)

    @patch('deepparse.network.embedding_network.torch.nn.Linear')
    def test_whenInstanciatingEmbeddingNetwork_thenShouldInstanciateLinearLayerWithCorrectParameters(self, linear_mock):
        self.embedding_network = EmbeddingNetwork(self.a_input_size, self.a_hidden_size, self.a_projection_size,
                                                  self.a_number_of_layers)

        linear_mock.assert_called_with(2 * self.a_hidden_size, self.a_projection_size)

    @patch('deepparse.network.embedding_network.torch.nn.MaxPool1d')
    def test_givenMaxpool_whenInstanciatingEmbeddingNetwork_thenShouldInstanciateMaxpoolLayerWithCorrectParameters(
            self, maxpool_mock):
        self.embedding_network = EmbeddingNetwork(self.a_input_size,
                                                  self.a_hidden_size,
                                                  self.a_projection_size,
                                                  self.a_number_of_layers,
                                                  maxpool=True,
                                                  maxpool_kernel_size=self.a_maxpool_kernel_size)

        maxpool_mock.asset_called_with(self.a_maxpool_kernel_size)


if __name__ == "__main__":
    unittest.main()
