# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import unittest
from unittest import TestCase

import torch

from deepparse.converter import (fasttext_data_padding, bpemb_data_padding, fasttext_data_padding_with_target,
                                 bpemb_data_padding_with_target, fasttext_data_padding_teacher_forcing,
                                 bpemb_data_padding_teacher_forcing)


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

        cls.a_training_non_padded_word_embedding_batch = []
        cls.a_training_non_padded_subword_embedding_batch = []
        cls.a_padded_target_tensor = []

    def setUp(self):
        self.a_non_padded_word_embedding_batch_length_list = torch.tensor([3, 2, 1])
        self.a_non_padded_word_embedding_batch = [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1]]]
        self.a_fasttext_padded_batch = torch.FloatTensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [-100, -100]],
                                                          [[1, 1], [-100, -100], [-100, -100]]])

        self.a_non_padded_subword_embedding_batch_lenght_list = torch.tensor([3, 2, 1])
        self.a_non_padded_subword_embedding_batch = [([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                       [[1, 1], [-1, -1], [-1, -1]]], [2, 3, 1]),
                                                     ([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]]], [2, 2]),
                                                     ([[[1, 1], [1, 1], [1, 1]]], [3])]
        self.a_bpemb_padded_batch = torch.tensor([[[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                   [[1, 1], [-1, -1], [-1, -1]]],
                                                  [[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]],
                                                   [[-100, -100], [-100, -100], [-100, -100]]],
                                                  [[[1, 1], [1, 1], [1, 1]], [[-100, -100], [-100, -100], [-100, -100]],
                                                   [[-100, -100], [-100, -100], [-100, -100]]]])
        self.a_non_padded_subword_embedding_batch_decomposition_lenght_list = [[2, 3, 1], [2, 2, 1], [3, 1, 1]]

        self.a_training_non_padded_word_embedding_batch = [([[1, 1], [1, 1], [1, 1]], [0, 3, 5]),
                                                           ([[1, 1], [1, 1]], [4, 7]), ([[1, 1]], [8])]

        self.a_training_non_padded_subword_embedding_batch = [(([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [1, 1]],
                                                                 [[1, 1], [-1, -1], [-1, -1]]], [2, 3, 1]), [0, 3, 5]),
                                                              (([[[1, 1], [1, 1], [-1, -1]],
                                                                 [[1, 1], [1, 1], [-1, -1]]], [2, 2]), [4, 7]),
                                                              (([[[1, 1], [1, 1], [1, 1]]], [3]), [8])]

        self.a_padded_target_tensor = torch.tensor([[0, 3, 5], [4, 7, -100], [8, -100, -100]])

        self.fasttext_data_padding = fasttext_data_padding
        self.bpemb_data_padding = bpemb_data_padding
        self.fasttext_data_padding_with_target = fasttext_data_padding_with_target
        self.bpemb_data_padding_with_target = bpemb_data_padding_with_target
        self.fasttext_data_padding_teacher_Forcing = fasttext_data_padding_teacher_forcing
        self.bpemb_data_padding_teacher_forcing = bpemb_data_padding_teacher_forcing

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

    def test_givenBatch_whenFasttextPaddingWithTarget_thenShouldReturnRightLengths(self):
        (_, lengths), _ = self.fasttext_data_padding_with_target(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_word_embedding_batch_length_list)))

    def test_givenBatch_whenFasttextPaddingWithTarget_thenShouldReturnBatchAsTensor(self):
        (padded_batch, _), _ = self.fasttext_data_padding_with_target(self.a_training_non_padded_word_embedding_batch)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenBatch_whenFasttextPaddingWithTarget_thenShouldPerformRightPadding(self):
        (padded_batch, _), _ = self.fasttext_data_padding_with_target(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(padded_batch.eq(self.a_fasttext_padded_batch)))

    def test_givenBatch_whenFasttextPaddingWithTarget_thenShouldReturnPaddedTarget(self):
        (_, _), target_tensor = self.fasttext_data_padding_with_target(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(target_tensor.eq(self.a_padded_target_tensor)))

    def test_givenBatch_whenBpembDataPaddingWithTarget_thenShouldReturnRightLengths(self):
        (_, _, lengths), _ = self.bpemb_data_padding_with_target(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_subword_embedding_batch_lenght_list)))

    def test_givenBatch_whenBpembDataPaddingWithTarget_thenShouldReturnBatchAsTensor(self):
        (padded_batch, _,
         _), _ = self.bpemb_data_padding_with_target(self.a_training_non_padded_subword_embedding_batch)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenBatch_whenBpembDataPaddingWithTarget_thenShouldPerformRightPadding(self):
        (padded_batch, _,
         _), _ = self.bpemb_data_padding_with_target(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(padded_batch.eq(self.a_bpemb_padded_batch)))

    def test_givenBatch_whenBpembDataPaddingWithTarget_thenShouldReturnRightDecompositionLengths(self):
        (_, decomposition_lengths,
         _), _ = self.bpemb_data_padding_with_target(self.a_training_non_padded_subword_embedding_batch)

        self.assertEqual(decomposition_lengths, self.a_non_padded_subword_embedding_batch_decomposition_lenght_list)

    def test_givenBatch_whenBpembDataPaddingWithTarget_thenShouldReturnPaddedTarget(self):
        (_, _,
         _), target_tensor = self.bpemb_data_padding_with_target(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(target_tensor.eq(self.a_padded_target_tensor)))

    def test_givenBatch_whenFasttextPaddingTeacherForcing_thenShouldReturnRightLengths(self):
        (_, lengths, _), _ = self.fasttext_data_padding_teacher_Forcing(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_word_embedding_batch_length_list)))

    def test_givenBatch_whenFasttextPaddingTeacherForcing_thenShouldReturnBatchAsTensor(self):
        (padded_batch, _,
         _), _ = self.fasttext_data_padding_teacher_Forcing(self.a_training_non_padded_word_embedding_batch)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenBatch_whenFasttextPaddingTeacherForcing_thenShouldPerformRightPadding(self):
        (padded_batch, _,
         _), _ = self.fasttext_data_padding_teacher_Forcing(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(padded_batch.eq(self.a_fasttext_padded_batch)))

    def test_givenBatch_whenFasttextPaddingTeacherForcing_thenShouldReturnPaddedTarget(self):
        (_, _,
         _), target_tensor = self.fasttext_data_padding_teacher_Forcing(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(target_tensor.eq(self.a_padded_target_tensor)))

    def test_givenBatch_whenFasttextPaddingTeacherForcing_thenShouldReturnPaddedTargetInBatch(self):
        (_, _,
         target_tensor), _ = self.fasttext_data_padding_teacher_Forcing(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(target_tensor.eq(self.a_padded_target_tensor)))

    def test_givenBatch_whenBpembDataPaddingTeacherForcing_thenShouldReturnRightLengths(self):
        (_, _, lengths,
         _), _ = self.bpemb_data_padding_teacher_forcing(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_subword_embedding_batch_lenght_list)))

    def test_givenBatch_whenBpembDataPaddingTeacherForcing_thenShouldReturnBatchAsTensor(self):
        (padded_batch, _, _,
         _), _ = self.bpemb_data_padding_teacher_forcing(self.a_training_non_padded_subword_embedding_batch)

        self.assertIsInstance(padded_batch, torch.Tensor)

    def test_givenBatch_whenBpembDataPaddingTeacherForcing_thenShouldPerformRightPadding(self):
        (padded_batch, _, _,
         _), _ = self.bpemb_data_padding_teacher_forcing(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(padded_batch.eq(self.a_bpemb_padded_batch)))

    def test_givenBatch_whenBpembDataPaddingTeacherForcing_thenShouldReturnRightDecompositionLengths(self):
        (_, decomposition_lengths, _,
         _), _ = self.bpemb_data_padding_teacher_forcing(self.a_training_non_padded_subword_embedding_batch)

        self.assertEqual(decomposition_lengths, self.a_non_padded_subword_embedding_batch_decomposition_lenght_list)

    def test_givenBatch_whenBpembDataPaddingTeacherForcing_thenShouldReturnPaddedTarget(self):
        (_, _, _,
         _), target_tensor = self.bpemb_data_padding_teacher_forcing(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(target_tensor.eq(self.a_padded_target_tensor)))

    def test_givenBatch_whenBpembDataPaddingTeacherForcing_thenShouldReturnTargetTensorInBatch(self):
        (_, _, _,
         target_tensor), _ = self.bpemb_data_padding_teacher_forcing(self.a_training_non_padded_subword_embedding_batch)

        self.assertTrue(torch.all(target_tensor.eq(self.a_padded_target_tensor)))


if __name__ == "__main__":
    unittest.main()
