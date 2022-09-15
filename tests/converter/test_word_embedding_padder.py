import unittest
from unittest import TestCase

import torch

from deepparse.converter import WordEmbeddingPadder


class WordEmbeddingPadderTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_padding_value = -100

    def setUp(self):
        self.a_non_padded_word_embedding_batch_length_list = torch.tensor([3, 2, 1])
        self.a_non_padded_word_embedding_sequences_batch = [
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 1]],
        ]
        self.a_padded_word_embedding_sequences_batch = torch.FloatTensor(
            [
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [-100, -100]],
                [[1, 1], [-100, -100], [-100, -100]],
            ]
        )

        self.a_training_non_padded_word_embedding_batch = [
            ([[1, 1], [1, 1], [1, 1]], [0, 3, 5]),
            ([[1, 1], [1, 1]], [4, 7]),
            ([[1, 1]], [8]),
        ]

        self.a_padded_target_tensor = torch.tensor([[0, 3, 5], [4, 7, -100], [8, -100, -100]])

        self.padder = WordEmbeddingPadder(self.a_padding_value)

    def test_givenASequencesBatch_whenPadding_thenShouldReturnCorrectLengths(self):
        _, lengths = self.padder.pad_sequences(self.a_non_padded_word_embedding_sequences_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_word_embedding_batch_length_list)))

    def test_givenASequencesBatch_whenPadding_thenShouldReturnBatchAsTensor(self):
        padded_sequences, _ = self.padder.pad_sequences(self.a_non_padded_word_embedding_sequences_batch)

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenASequencesBatch_whenPadding_thenShouldPerformCorrectPadding(self):
        padded_sequences, _ = self.padder.pad_sequences(self.a_non_padded_word_embedding_sequences_batch)

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_word_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPadding_thenShouldReturnCorrectLengths(self):
        (_, lengths), _ = self.padder.pad_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_word_embedding_batch_length_list)))

    def test_givenATrainingBatch_whenPadding_thenShouldReturnBatchAsTensor(self):
        (padded_sequences, _), _ = self.padder.pad_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenATrainingBatch_whenPadding_thenShouldPerformCorrectPadding(self):
        (padded_sequences, _), _ = self.padder.pad_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_word_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPadding_thenShouldReturnTargetAsTensor(self):
        (_, _), padded_target = self.padder.pad_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertIsInstance(padded_target, torch.Tensor)

    def test_givenATrainingBatch_whenPadding_thenShouldPerformCorrectPaddingOnTarget(self):
        (_, _), padded_target = self.padder.pad_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenATrainingBatch_whenPaddingWithTeacherForcing_thenShouldReturnCorrectLengths(self):
        (_, lengths, _), _ = self.padder.pad_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(lengths.eq(self.a_non_padded_word_embedding_batch_length_list)))

    def test_givenATrainingBatch_whenPaddingWithTeacherForcing_thenShouldReturnBatchAsTensor(self):
        (padded_sequences, _, _), _ = self.padder.pad_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingWithTeacherForcing_thenShouldPerformCorrectPadding(self):
        (padded_sequences, _, _), _ = self.padder.pad_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_word_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingWithTeacherForcing_thenShouldReturnTargetAsTensor(self):
        (_, _, _), padded_target = self.padder.pad_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertIsInstance(padded_target, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingWithTeacherForcing_thenShouldPerformCorrectPaddingOnTarget(self):
        (_, _, _), padded_target = self.padder.pad_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenATrainingBatch_whenPaddingWithTeacherForcing_thenShouldReturnTargetWithSequencesAndLengths(self):
        (_, _, padded_target), _ = self.padder.pad_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))


if __name__ == "__main__":
    unittest.main()
