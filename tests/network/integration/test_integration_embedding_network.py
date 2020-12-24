# pylint: disable=line-too-long
import unittest
from unittest import TestCase

import torch

from deepparse.network import EmbeddingNetwork


class EmbeddingNetworkTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_input_size = 4
        cls.a_hidden_size = 2
        cls.a_projection_size = 4

        cls.a_subword_embeddings_tensor = torch.Tensor([[[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                         [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                         [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]],
                                                        [[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                         [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                         [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]]])

        cls.a_decomposition_lengths_tuple = ([3, 1, 2], [1, 3, 2])

        cls.a_max_sequence_length = 3
        cls.a_batch_size = 2
        cls.a_embedding_dimension = 4
        cls.a_post_maxpool_embedding_dimension = 2

    def test_givenASubwordEmbeddingsTensor_whenCallingEmbeddingNetwork_thenShouldReturnTensorWithCorrectDimension(self):
        self.embedding_network = EmbeddingNetwork(self.a_input_size, self.a_hidden_size, self.a_projection_size)

        result = self.embedding_network(self.a_subword_embeddings_tensor, self.a_decomposition_lengths_tuple)

        self.assertEqual(result.size(0), self.a_batch_size)
        self.assertEqual(result.size(1), self.a_max_sequence_length)
        self.assertEqual(result.size(2), self.a_embedding_dimension)

    def test_givenASubwordEmbeddingsTensorAndMaxPool_whenCallingEmbeddingNetwork_thenShouldReturnTensorWithCorrectDimension(
            self):
        self.embedding_network = EmbeddingNetwork(self.a_input_size,
                                                  self.a_hidden_size,
                                                  self.a_projection_size,
                                                  maxpool=True,
                                                  maxpool_kernel_size=2)

        result = self.embedding_network(self.a_subword_embeddings_tensor, self.a_decomposition_lengths_tuple)

        self.assertEqual(result.size(0), self.a_batch_size)
        self.assertEqual(result.size(1), self.a_max_sequence_length)
        self.assertEqual(result.size(2), self.a_post_maxpool_embedding_dimension)


if __name__ == "__main__":
    unittest.main()
