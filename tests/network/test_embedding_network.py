import unittest
from typing import List
from unittest import TestCase
from unittest.mock import patch

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from deepparse.network import EmbeddingNetwork


def reference_forward(network: EmbeddingNetwork, to_predict: torch.Tensor, decomposition_lengths: List) -> torch.Tensor:
    # The former implementation of EmbeddingNetwork.forward (one LSTM call per word position), kept as the reference
    # of the expected output.
    batch_size = to_predict.size(0)
    embeddings = torch.zeros(
        to_predict.size(1), to_predict.size(0), int(to_predict.size(3) / network.maxpool_kernel_size)
    )
    to_predict = to_predict.transpose(0, 1).float()
    for i in range(to_predict.size(0)):
        lengths = [decomposition_length[i] for decomposition_length in decomposition_lengths]
        packed_sequence = pack_padded_sequence(to_predict[i], lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = network.model(packed_sequence)
        padded_output, padded_output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        word_context = torch.zeros(padded_output.size(0), padded_output.size(2))
        for j in range(batch_size):
            word_context[j] = padded_output[j, padded_output_lengths[j] - 1, :]
        projection_output = network.projection_layer(word_context)
        if network.maxpooling_layer is not None:
            projection_output = network._max_pool(projection_output)  # pylint: disable=protected-access
        embeddings[i] = projection_output
    return embeddings.transpose(0, 1)


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

    @patch("deepparse.network.embedding_network.torch.nn.LSTM")
    def test_whenInstanciatingEmbeddingNetwork_thenShouldInstanciateLSTMWithCorrectParameters(self, lstm_mock):
        self.embedding_network = EmbeddingNetwork(
            self.a_input_size,
            self.a_hidden_size,
            self.a_projection_size,
            self.a_number_of_layers,
        )

        lstm_mock.assert_called_with(
            self.a_input_size,
            self.a_hidden_size,
            num_layers=self.a_number_of_layers,
            batch_first=True,
            bidirectional=True,
        )

    @patch("deepparse.network.embedding_network.torch.nn.Linear")
    def test_whenInstanciatingEmbeddingNetwork_thenShouldInstanciateLinearLayerWithCorrectParameters(self, linear_mock):
        self.embedding_network = EmbeddingNetwork(
            self.a_input_size,
            self.a_hidden_size,
            self.a_projection_size,
            self.a_number_of_layers,
        )

        linear_mock.assert_called_with(2 * self.a_hidden_size, self.a_projection_size)

    @patch("deepparse.network.embedding_network.torch.nn.MaxPool1d")
    def test_givenMaxpool_whenInstanciatingEmbeddingNetwork_thenShouldInstanciateMaxpoolLayerWithCorrectParameters(
        self, maxpool_mock
    ):
        self.embedding_network = EmbeddingNetwork(
            self.a_input_size,
            self.a_hidden_size,
            self.a_projection_size,
            self.a_number_of_layers,
            maxpool=True,
            maxpool_kernel_size=self.a_maxpool_kernel_size,
        )

        maxpool_mock.asset_called_with(self.a_maxpool_kernel_size)


class EmbeddingNetworkForwardTest(TestCase):
    # Real (not mocked) forward passes: the batched forward must give the same output as the former per-word loop.

    def setUp(self):
        torch.manual_seed(42)
        self.embeddings_dimension = 12
        # Three addresses padded to 4 words of at most 5 sub-words; the padding words have a decomposition length of 1.
        self.decomposition_lengths = [[2, 5, 1, 3], [1, 1, 1, 1], [4, 2, 1, 1]]
        self.to_predict = torch.rand(3, 4, 5, self.embeddings_dimension)

    def assert_same_output_as_reference(self, network: EmbeddingNetwork) -> None:
        with torch.no_grad():
            expected = reference_forward(network, self.to_predict, self.decomposition_lengths)
            actual = network(self.to_predict, self.decomposition_lengths)

        self.assertEqual(expected.shape, actual.shape)
        torch.testing.assert_close(actual, expected)

    def test_givenNoMaxpool_whenForward_thenSameOutputAsPerWordLoop(self):
        network = EmbeddingNetwork(self.embeddings_dimension, 6, self.embeddings_dimension)

        self.assert_same_output_as_reference(network)

    def test_givenMaxpool_whenForward_thenSameOutputAsPerWordLoop(self):
        network = EmbeddingNetwork(self.embeddings_dimension, 6, self.embeddings_dimension, maxpool=True)

        self.assert_same_output_as_reference(network)

    def test_givenTwoLayers_whenForward_thenSameOutputAsPerWordLoop(self):
        network = EmbeddingNetwork(self.embeddings_dimension, 6, self.embeddings_dimension, num_layers=2)

        self.assert_same_output_as_reference(network)

    def test_whenForward_thenOutputDependsOnlyOnTheValidSubWords(self):
        network = EmbeddingNetwork(self.embeddings_dimension, 6, self.embeddings_dimension)
        noisy_padding = self.to_predict.clone()
        # Sub-words past each word decomposition length are padding: changing them must not change the output.
        for address_index, address_lengths in enumerate(self.decomposition_lengths):
            for word_index, length in enumerate(address_lengths):
                noisy_padding[address_index, word_index, length:] = 100.0

        with torch.no_grad():
            torch.testing.assert_close(
                network(noisy_padding, self.decomposition_lengths),
                network(self.to_predict, self.decomposition_lengths),
            )

    def test_whenBackward_thenSameGradientsAsPerWordLoop(self):
        network = EmbeddingNetwork(self.embeddings_dimension, 6, self.embeddings_dimension)

        reference_forward(network, self.to_predict, self.decomposition_lengths).sum().backward()
        expected_gradients = [parameter.grad.clone() for parameter in network.parameters()]
        network.zero_grad()
        network(self.to_predict, self.decomposition_lengths).sum().backward()

        for actual, expected in zip((parameter.grad for parameter in network.parameters()), expected_gradients):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
