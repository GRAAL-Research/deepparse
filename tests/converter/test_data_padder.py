# pylint: disable=line-too-long, too-many-public-methods
import unittest
from unittest import TestCase

import torch

from deepparse.converter import DataPadder


class DataPadderTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a_padding_value = -100

    def setUp(self):
        self.a_non_padded_word_embedding_batch_length_list = [3, 2, 1]
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

        self.a_non_padded_subword_embedding_batch_length_list = [3, 2, 1]
        self.a_non_padded_subword_embedding_sequences_batch = [
            (
                [
                    [[1, 1], [1, 1], [-1, -1]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1], [-1, -1], [-1, -1]],
                ],
                [2, 3, 1],
            ),
            ([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]]], [2, 2]),
            ([[[1, 1], [1, 1], [1, 1]]], [3]),
        ]

        self.a_padded_subword_embedding_sequences_batch = torch.tensor(
            [
                [
                    [[1, 1], [1, 1], [-1, -1]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1], [-1, -1], [-1, -1]],
                ],
                [
                    [[1, 1], [1, 1], [-1, -1]],
                    [[1, 1], [1, 1], [-1, -1]],
                    [[-100, -100], [-100, -100], [-100, -100]],
                ],
                [
                    [[1, 1], [1, 1], [1, 1]],
                    [[-100, -100], [-100, -100], [-100, -100]],
                    [[-100, -100], [-100, -100], [-100, -100]],
                ],
            ]
        )
        self.a_non_padded_subword_embedding_batch_decomposition_length_list = [
            [2, 3, 1],
            [2, 2, 1],
            [3, 1, 1],
        ]

        self.a_training_non_padded_subword_embedding_batch = [
            (
                (
                    [
                        [[1, 1], [1, 1], [-1, -1]],
                        [[1, 1], [1, 1], [1, 1]],
                        [[1, 1], [-1, -1], [-1, -1]],
                    ],
                    [2, 3, 1],
                ),
                [0, 3, 5],
            ),
            (
                ([[[1, 1], [1, 1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]]], [2, 2]),
                [4, 7],
            ),
            (([[[1, 1], [1, 1], [1, 1]]], [3]), [8]),
        ]

        self.a_padded_target_tensor = torch.tensor([[0, 3, 5], [4, 7, -100], [8, -100, -100]])

        self.padder = DataPadder(self.a_padding_value)

    def test_givenASequencesBatch_whenPaddingWordEmbeddings_thenShouldReturnCorrectLengths(self):
        _, lengths = self.padder.pad_word_embeddings_sequences(self.a_non_padded_word_embedding_sequences_batch)

        self.assertEqual(lengths, self.a_non_padded_word_embedding_batch_length_list)

    def test_givenASequencesBatch_whenPaddingWordEmbeddings_thenShouldReturnBatchAsTensor(self):
        padded_sequences, _ = self.padder.pad_word_embeddings_sequences(
            self.a_non_padded_word_embedding_sequences_batch
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenASequencesBatch_whenPaddingWordEmbeddings_thenShouldPerformCorrectPadding(self):
        padded_sequences, _ = self.padder.pad_word_embeddings_sequences(
            self.a_non_padded_word_embedding_sequences_batch
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_word_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingWordEmbeddings_thenShouldReturnCorrectLengths(self):
        (_, lengths), _ = self.padder.pad_word_embeddings_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertEqual(lengths, self.a_non_padded_word_embedding_batch_length_list)

    def test_givenATrainingBatch_whenPaddingWordEmbeddings_thenShouldReturnBatchAsTensor(self):
        (padded_sequences, _), _ = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingWordEmbeddings_thenShouldPerformCorrectPadding(self):
        (padded_sequences, _), _ = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_word_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingWordEmbeddings_thenShouldReturnTargetAsTensor(self):
        (_, _), padded_target = self.padder.pad_word_embeddings_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertIsInstance(padded_target, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingWordEmbeddings_thenShouldPerformCorrectPaddingOnTarget(self):
        (_, _), padded_target = self.padder.pad_word_embeddings_batch(self.a_training_non_padded_word_embedding_batch)

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenATrainingBatch_whenPaddingWordEmbeddingsWithTeacherForcing_thenShouldReturnCorrectLengths(self):
        (_, lengths, _), _ = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertEqual(lengths, self.a_non_padded_word_embedding_batch_length_list)

    def test_givenATrainingBatch_whenPaddingWordEmbeddingsWithTeacherForcing_thenShouldReturnBatchAsTensor(self):
        (padded_sequences, _, _), _ = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingWordEmbeddingsWithTeacherForcing_thenShouldPerformCorrectPadding(self):
        (padded_sequences, _, _), _ = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_word_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingWordEmbeddingsWithTeacherForcing_thenShouldReturnTargetAsTensor(self):
        (_, _, _), padded_target = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertIsInstance(padded_target, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingWordEmbeddingsWithTeacherForcing_thenShouldPerformCorrectPaddingOnTarget(
        self,
    ):
        (_, _, _), padded_target = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenATrainingBatch_whenPaddingWordEmbeddingsWithTeacherForcing_thenShouldReturnTargetWithSequencesAndLengths(
        self,
    ):
        (_, _, padded_target), _ = self.padder.pad_word_embeddings_batch(
            self.a_training_non_padded_word_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenASequencesBatch_whenPaddingSubwordEmbeddings_thenShouldReturnCorrectLengths(self):
        _, _, lengths = self.padder.pad_subword_embeddings_sequences(
            self.a_non_padded_subword_embedding_sequences_batch
        )

        self.assertEqual(lengths, self.a_non_padded_subword_embedding_batch_length_list)

    def test_givenASequencesBatch_whenPaddingSubwordEmbeddings_thenShouldReturnCorrectDecompositionLengths(self):
        _, decomposition_lengths, _ = self.padder.pad_subword_embeddings_sequences(
            self.a_non_padded_subword_embedding_sequences_batch
        )

        self.assertEqual(decomposition_lengths, self.a_non_padded_subword_embedding_batch_decomposition_length_list)

    def test_givenASequencesBatch_whenPaddingSubwordEmbeddings_thenShouldReturnBatchAsTensor(self):
        padded_sequences, _, _ = self.padder.pad_subword_embeddings_sequences(
            self.a_non_padded_subword_embedding_sequences_batch
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenASequencesBatch_whenPaddingSubwordEmbeddings_thenShouldPerformCorrectPadding(self):
        padded_sequences, _, _ = self.padder.pad_subword_embeddings_sequences(
            self.a_non_padded_subword_embedding_sequences_batch
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_subword_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddings_thenShouldReturnCorrectLengths(self):
        (_, _, lengths), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch
        )

        self.assertEqual(lengths, self.a_non_padded_subword_embedding_batch_length_list)

    def test_givenATrainingsBatch_whenPaddingSubwordEmbeddings_thenShouldReturnCorrectDecompositionLengths(self):
        (_, decomposition_lengths, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch
        )

        self.assertEqual(decomposition_lengths, self.a_non_padded_subword_embedding_batch_decomposition_length_list)

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddings_thenShouldReturnBatchAsTensor(self):
        (padded_sequences, _, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddings_thenShouldPerformCorrectPadding(self):
        (padded_sequences, _, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_subword_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddings_thenShouldReturnTargetAsTensor(self):
        (_, _, _), padded_target = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch
        )

        self.assertIsInstance(padded_target, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddings_thenShouldPerformCorrectPaddingOnTarget(self):
        (_, _, _), padded_target = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldReturnCorrectLengths(self):
        (_, _, lengths, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertEqual(lengths, self.a_non_padded_subword_embedding_batch_length_list)

    def test_givenATrainingsBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldReturnCorrectDecompositionLengths(
        self,
    ):
        (_, decomposition_lengths, _, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertEqual(decomposition_lengths, self.a_non_padded_subword_embedding_batch_decomposition_length_list)

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldReturnBatchAsTensor(self):
        (padded_sequences, _, _, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertIsInstance(padded_sequences, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldPerformCorrectPadding(self):
        (padded_sequences, _, _, _), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_sequences.eq(self.a_padded_subword_embedding_sequences_batch)))

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldReturnTargetAsTensor(self):
        (_, _, _, _), padded_target = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertIsInstance(padded_target, torch.Tensor)

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldPerformCorrectPaddingOnTarget(
        self,
    ):
        (_, _, _, _), padded_target = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))

    def test_givenATrainingBatch_whenPaddingSubwordEmbeddingsWithTeacherForcing_thenShouldReturnTargetWithSequencesAndLengths(
        self,
    ):
        (_, _, _, padded_target), _ = self.padder.pad_subword_embeddings_batch(
            self.a_training_non_padded_subword_embedding_batch, teacher_forcing=True
        )

        self.assertTrue(torch.all(padded_target.eq(self.a_padded_target_tensor)))


if __name__ == "__main__":
    unittest.main()
