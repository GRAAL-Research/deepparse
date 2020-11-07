# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

import unittest
from unittest import TestCase

import torch

from deepparse.converter import data_padding, bpemb_data_padding


class DataPaddingTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_number_of_sequences = 5
        cls.a_max_length = 10

        cls.a_non_padded_word_embedding_batch_length_list = []
        cls.a_non_padded_word_embedding_batch = []
        cls.a_fasttext_padded_batch = []

        cls.a_non_padded_subword_embedding_batch_lenght_list = []
        cls.a_non_padded_subword_embedding_batch_decomposition_lenght_list = []
        cls.a_non_padded_subword_embedding_batch = []
        cls.a_bpemb_padded_batch = []

    def setUp(self):
        self.a_non_padded_word_embedding_batch_length_list = torch.tensor([3, 2, 1])
        self.a_non_padded_word_embedding_batch = [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1]]]
        self.a_fasttext_padded_batch = torch.FloatTensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [0, 0]],
                                                          [[1, 1], [0, 0], [0, 0]]])

        self.a_non_padded_subword_embedding_batch_lenght_list = torch.tensor([3, 2, 1])
        self.a_non_padded_subword_embedding_batch = [([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                       [[1, 1], [-1, -1], [-1, -1]]], [2, 3, 1]),
                                                     ([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]]], [2, 2]),
                                                     ([[[1, 1], [1, 1], [1, 1]]], [3])]
        self.a_bpemb_padded_batch = torch.tensor([[[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                   [[1, 1], [-1, -1], [-1, -1]]],
                                                  [[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]],
                                                   [[0, 0], [0, 0], [0, 0]]],
                                                  [[[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0]],
                                                   [[0, 0], [0, 0], [0, 0]]]])
        self.a_non_padded_subword_embedding_batch_decomposition_lenght_list = [[2, 3, 1], [2, 2, 1], [3, 1, 1]]

        self.fasttext_data_padding = data_padding
        self.bpemb_data_padding = bpemb_data_padding

    def test_givenbatch_whenfasttextpadding_thenshouldreturnrightlengths(self):
        _, lengths = self.fasttext_data_padding(self.a_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_word_embedding_batch_length_list)))

    def test_whenfasttextpadding_thenshouldreturnbatchastensor(self):
        padded_batch, _ = self.fasttext_data_padding(self.a_non_padded_word_embedding_batch)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenbatch_whenfasttextpadding_thenshoulddorightpadding(self):
        padded_batch, _ = self.fasttext_data_padding(self.a_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(padded_batch.eq(self.a_fasttext_padded_batch)))

    def test_givenbatch_whenbpembpadding_thenshouldreturnrightlengths(self):
        _, _, lengths = self.bpemb_data_padding(self.a_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_subword_embedding_batch_lenght_list)))

    def test_givenbatch_whenbpembpadding_thenshouldreturnrightdecomposition_lengths(self):
        _, decomposition_lengths, _ = self.bpemb_data_padding(self.a_non_padded_subword_embedding_batch)

        self.assertEqual(decomposition_lengths, self.a_non_padded_subword_embedding_batch_decomposition_lenght_list)

    def test_whenbpembpadding_thenshouldreturnbatchastensor(self):
        padded_batch, _, _ = self.bpemb_data_padding(self.a_non_padded_subword_embedding_batch)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenbatch_whenbpembpadding_thenshoulddorightpadding(self):
        padded_batch, _, _ = self.bpemb_data_padding(self.a_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(padded_batch.eq(self.a_bpemb_padded_batch)))


if __name__ == '__main__':
    unittest.main()
