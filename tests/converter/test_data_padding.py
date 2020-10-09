import unittest
from unittest import TestCase
import numpy as np
import torch

from deepparse.converter import data_padding, bpemb_data_padding


class DataPaddingTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A_NUMBER_OF_SEQUENCES = 5
        cls.A_MAX_LENGTH = 10

        cls.A_NON_PADDED_WORD_EMBEDDING_BATCH_LENGTH_LIST = []
        cls.A_NON_PADDED_WORD_EMBEDDING_BATCH = []
        cls.A_FASTTEXT_PADDED_BATCH = []

        cls.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH_LENGHT_LIST = []
        cls.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH_DECOMPOSITION_LENGHT_LIST = []
        cls.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH = []
        cls.A_BPEMB_PADDED_BATCH = []

    def setUp(self):
        self.A_NON_PADDED_WORD_EMBEDDING_BATCH_LENGTH_LIST = torch.tensor([3, 2, 1])
        self.A_NON_PADDED_WORD_EMBEDDING_BATCH = [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1]]]
        self.A_FASTTEXT_PADDED_BATCH = torch.FloatTensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [0, 0]],
                                                          [[1, 1], [0, 0], [0, 0]]])

        self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH_LENGHT_LIST = torch.tensor([3, 2, 1])
        self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH = [([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                       [[1, 1], [-1, -1], [-1, -1]]], [2, 3, 1]),
                                                     ([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]]], [2, 2]),
                                                     ([[[1, 1], [1, 1], [1, 1]]], [3])]
        self.A_BPEMB_PADDED_BATCH = torch.tensor([[[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                   [[1, 1], [-1, -1], [-1, -1]]],
                                                  [[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]],
                                                   [[0, 0], [0, 0], [0, 0]]],
                                                  [[[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0]],
                                                   [[0, 0], [0, 0], [0, 0]]]])
        self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH_DECOMPOSITION_LENGHT_LIST = [[2, 3, 1], [2, 2, 1], [3, 1, 1]]

        self.fasttext_data_padding = data_padding
        self.bpemb_data_padding = bpemb_data_padding

    def test_givenBatch_whenFasttextPadding_thenShouldReturnRightLengths(self):
        _, lengths = self.fasttext_data_padding(self.A_NON_PADDED_WORD_EMBEDDING_BATCH)

        self.assertTrue(torch.all(lengths.eq(self.A_NON_PADDED_WORD_EMBEDDING_BATCH_LENGTH_LIST)))

    def test_whenFasttextPadding_thenShouldReturnBatchAsTensor(self):
        padded_batch, _ = self.fasttext_data_padding(self.A_NON_PADDED_WORD_EMBEDDING_BATCH)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenBatch_whenFasttextPadding_thenShouldDoRightPadding(self):
        padded_batch, _ = self.fasttext_data_padding(self.A_NON_PADDED_WORD_EMBEDDING_BATCH)

        self.assertTrue(torch.all(padded_batch.eq(self.A_FASTTEXT_PADDED_BATCH)))

    def test_givenBatch_whenBpembPadding_thenShouldReturnRightLengths(self):
        _, _, lengths = self.bpemb_data_padding(self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH)

        self.assertTrue(torch.all(lengths.eq(self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH_LENGHT_LIST)))

    
    def test_givenBatch_whenBpembPadding_thenShouldReturnRightDecomposition_Lengths(self):
        _, decomposition_lengths, _ = self.bpemb_data_padding(self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH)

        self.assertEqual(decomposition_lengths, self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH_DECOMPOSITION_LENGHT_LIST)

    def test_whenBpembPadding_thenShouldReturnBatchAsTensor(self):
        padded_batch, _, _ = self.bpemb_data_padding(self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenBatch_whenBpembPadding_thenShouldDoRightPadding(self):
        padded_batch, _, _ = self.bpemb_data_padding(self.A_NON_PADDED_SUBWORD_EMBEDDING_BATCH)

        self.assertTrue(torch.all(padded_batch.eq(self.A_BPEMB_PADDED_BATCH)))


if __name__ == '__main__':
    unittest.main()
