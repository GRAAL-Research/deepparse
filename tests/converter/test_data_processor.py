# pylint: disable=line-too-long
import unittest
from unittest import TestCase
from unittest.mock import ANY, MagicMock, Mock, call

import torch

from deepparse.converter.data_processor import DataProcessor


class DataProcessorTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an_address_list = ["45 old road", "quebec g1v0a1"]

        cls.a_tag_list = [["StreetNumber", "StreetName", "StreetName"], ["Municipality", "PostalCode"]]
        cls.a_address_and_tags_list = list(zip(cls.an_address_list, cls.a_tag_list))

        cls.a_word_embedding_sequence = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]]
        cls.a_padded_word_embedding_sequence = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [-100, -100]]])
        cls.a_sequence_lengths_list = torch.tensor([3, 2])

        cls.a_subword_embedding_sequence = [
            [[[1, 2], [3, 4], [-1, -1]], [[5, 6], [7, 8], [9, 10]], [[11, 12], [-1, -1], [-1, -1]]],
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [-1, -1], [-1, -1]]],
        ]
        cls.a_padded_subword_embedding_sequence = torch.tensor(
            [
                [[[1, 2], [3, 4], [-1, -1]], [[5, 6], [7, 8], [9, 10]], [[11, 12], [-1, -1], [-1, -1]]],
                [
                    [[13, 14], [15, 16], [17, 18]],
                    [[19, 20], [-1, -1], [-1, -1]],
                    [[-100, -100], [-100, -100], [-100, -100]],
                ],
            ]
        )
        cls.a_word_decomposition_lengths_list = [[2, 3, 1], [3, 1]]

        cls.a_subword_vectorized_sequence = list(
            zip(cls.a_subword_embedding_sequence, cls.a_word_decomposition_lengths_list)
        )

        cls.a_tag_to_idx = {"StreetNumber": 0, "StreetName": 1, "Municipality": 2, "PostalCode": 3, "EOS": 4}

        cls.a_tag_targets_list = [[0, 1, 1, 4], [2, 3, 4]]
        cls.a_padded_tag_targets = torch.tensor([[0, 1, 1, 4], [2, 3, 4, -100]])

    def setUp(self):
        self.fasttext_vectorizer_mock = MagicMock(return_value=self.a_word_embedding_sequence)
        self.bpemb_vectorizer_mock = MagicMock(return_value=self.a_subword_vectorized_sequence)

        self.fasttext_sequences_padding_callback_mock = Mock(
            return_value=(
                self.a_padded_word_embedding_sequence,
                self.a_sequence_lengths_list,
            )
        )
        self.bpemb_sequences_padding_callback_mock = Mock(
            return_value=(
                self.a_padded_subword_embedding_sequence,
                self.a_word_decomposition_lengths_list,
                self.a_sequence_lengths_list,
            )
        )

        self.fasttext_batch_padding_callback_mock = Mock()
        self.fasttext_batch_padding_callback_mock.side_effect = (
            lambda *params: (
                (
                    self.a_padded_word_embedding_sequence,
                    self.a_sequence_lengths_list,
                ),
                self.a_padded_tag_targets,
            )
            if params[1] is False
            else (
                (self.a_padded_word_embedding_sequence, self.a_sequence_lengths_list, self.a_padded_tag_targets),
                self.a_padded_tag_targets,
            )
        )

        self.bpemb_batch_padding_callback_mock = Mock(
            return_value=(
                (
                    self.a_padded_subword_embedding_sequence,
                    self.a_word_decomposition_lengths_list,
                    self.a_sequence_lengths_list,
                ),
                self.a_padded_tag_targets,
            )
        )
        self.bpemb_batch_padding_callback_mock.side_effect = (
            lambda *params: (
                (
                    self.a_padded_subword_embedding_sequence,
                    self.a_word_decomposition_lengths_list,
                    self.a_sequence_lengths_list,
                ),
                self.a_padded_tag_targets,
            )
            if params[1] is False
            else (
                (
                    self.a_padded_subword_embedding_sequence,
                    self.a_word_decomposition_lengths_list,
                    self.a_sequence_lengths_list,
                    self.a_padded_tag_targets,
                ),
                self.a_padded_tag_targets,
            )
        )

        self.tags_converter_mock = Mock()
        self.tags_converter_mock.side_effect = lambda tag: self.a_tag_to_idx[tag]

    def test_whenProcessingForInference_thenShouldCallVectorizerWithAddresses(self):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_inference(self.an_address_list)

        self.fasttext_vectorizer_mock.assert_called_once_with(self.an_address_list)

    def test_givenAFasttextEmbeddingContext_whenProcessingForInference_thenShouldCallSequencesPaddingCallbackWithCorrectEmbeddings(
        self,
    ):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_inference(self.an_address_list)

        self.fasttext_sequences_padding_callback_mock.assert_called_once_with(self.a_word_embedding_sequence)

    def test_givenAFasttextEmbeddingContext_whenProcessingForInference_thenShouldReturnCorrectPaddedEmbeddingSequences(
        self,
    ):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        sequences, lengths = processor.process_for_inference(self.an_address_list)

        self.assertTrue(torch.all(sequences.eq(self.a_padded_word_embedding_sequence)))
        self.assertTrue(torch.all(lengths.eq(self.a_sequence_lengths_list)))

    def test_givenABpembEmbeddingContext_whenProcessingForInference_thenShouldCallSequencesPaddingCallbackWithCorrectEmbeddings(
        self,
    ):
        processor = DataProcessor(
            self.bpemb_vectorizer_mock,
            self.bpemb_sequences_padding_callback_mock,
            self.bpemb_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_inference(self.an_address_list)

        self.bpemb_sequences_padding_callback_mock.assert_called_once_with(self.a_subword_vectorized_sequence)

    def test_givenABpembEmbeddingContext_whenProcessingForInference_thenShouldReturnCorrectPaddedEmbeddingSequences(
        self,
    ):
        processor = DataProcessor(
            self.bpemb_vectorizer_mock,
            self.bpemb_sequences_padding_callback_mock,
            self.bpemb_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        sequences, word_decomposition_lengths_list, lengths = processor.process_for_inference(self.an_address_list)

        self.assertTrue(torch.all(sequences.eq(self.a_padded_subword_embedding_sequence)))
        self.assertEqual(word_decomposition_lengths_list, self.a_word_decomposition_lengths_list)
        self.assertTrue(torch.all(lengths.eq(self.a_sequence_lengths_list)))

    def test_whenProcessingForTraining_thenShouldCallVectorizerWithAddresses(self):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_training(self.a_address_and_tags_list)

        self.fasttext_vectorizer_mock.assert_called_once_with(self.an_address_list)

    def test_givenAFasttextEmbeddingContext_whenProcessingForTraining_thenShouldCallBatchPaddingCallbackWithCorrectEmbeddings(
        self,
    ):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_training(self.a_address_and_tags_list)

        self.fasttext_batch_padding_callback_mock.assert_called_once_with(
            list(zip(self.a_word_embedding_sequence, self.a_tag_targets_list)), ANY
        )

    def test_givenAFasttextEmbeddingContext_whenProcessingForTraining_thenShouldReturnCorrectPaddedEmbeddingSequencesAndTargets(
        self,
    ):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        (sequences, lengths), targets = processor.process_for_training(self.a_address_and_tags_list)

        self.assertTrue(torch.all(sequences.eq(self.a_padded_word_embedding_sequence)))
        self.assertTrue(torch.all(lengths.eq(self.a_sequence_lengths_list)))
        self.assertTrue(torch.all(targets.eq(self.a_padded_tag_targets)))

    def test_givenABpembEmbeddingContext_whenProcessingForTraining_thenShouldCallBatchPaddingCallbackWithCorrectEmbeddings(
        self,
    ):
        processor = DataProcessor(
            self.bpemb_vectorizer_mock,
            self.bpemb_sequences_padding_callback_mock,
            self.bpemb_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_training(self.a_address_and_tags_list)

        self.bpemb_batch_padding_callback_mock.assert_called_once_with(
            list(zip(self.a_subword_vectorized_sequence, self.a_tag_targets_list)), ANY
        )

    def test_givenABpembEmbeddingContext_whenProcessingForTraining_thenShouldReturnCorrectPaddedEmbeddingSequencesAndTargets(
        self,
    ):
        processor = DataProcessor(
            self.bpemb_vectorizer_mock,
            self.bpemb_sequences_padding_callback_mock,
            self.bpemb_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        (sequences, word_decomposition_lengths, lengths), targets = processor.process_for_training(
            self.a_address_and_tags_list
        )

        self.assertTrue(torch.all(sequences.eq(self.a_padded_subword_embedding_sequence)))
        self.assertEqual(word_decomposition_lengths, self.a_word_decomposition_lengths_list)
        self.assertTrue(torch.all(lengths.eq(self.a_sequence_lengths_list)))
        self.assertTrue(torch.all(targets.eq(self.a_padded_tag_targets)))

    def test_whenProcessingForTraining_thenShouldCallTagsConverterToConvertTags(self):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        processor.process_for_training(self.a_address_and_tags_list)

        tags_converter_calls = [call(tag) for tags in self.a_tag_list for tag in tags + ["EOS"]]
        self.tags_converter_mock.assert_has_calls(tags_converter_calls)

    def test_givenAFasttextEmbeddingContext_whenProcessingForTrainingWithTeacherForcing_thenShouldReturnCorrectPaddedEmbeddingSequencesAndTargets(
        self,
    ):
        processor = DataProcessor(
            self.fasttext_vectorizer_mock,
            self.fasttext_sequences_padding_callback_mock,
            self.fasttext_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        (sequences, lengths, targets_input), targets = processor.process_for_training(
            self.a_address_and_tags_list, teacher_forcing=True
        )

        self.assertTrue(torch.all(sequences.eq(self.a_padded_word_embedding_sequence)))
        self.assertTrue(torch.all(lengths.eq(self.a_sequence_lengths_list)))
        self.assertTrue(torch.all(targets_input.eq(self.a_padded_tag_targets)))
        self.assertTrue(torch.all(targets.eq(self.a_padded_tag_targets)))

    def test_givenABpembEmbeddingContext_whenProcessingForTrainingWithTeacherForcing_thenShouldReturnCorrectPaddedEmbeddingSequencesAndTargets(
        self,
    ):
        processor = DataProcessor(
            self.bpemb_vectorizer_mock,
            self.bpemb_sequences_padding_callback_mock,
            self.bpemb_batch_padding_callback_mock,
            self.tags_converter_mock,
        )

        (sequences, word_decomposition_lengths, lengths, targets_input), targets = processor.process_for_training(
            self.a_address_and_tags_list, teacher_forcing=True
        )

        self.assertTrue(torch.all(sequences.eq(self.a_padded_subword_embedding_sequence)))
        self.assertEqual(word_decomposition_lengths, self.a_word_decomposition_lengths_list)
        self.assertTrue(torch.all(lengths.eq(self.a_sequence_lengths_list)))
        self.assertTrue(torch.all(targets_input.eq(self.a_padded_tag_targets)))
        self.assertTrue(torch.all(targets.eq(self.a_padded_tag_targets)))


if __name__ == "__main__":
    unittest.main()
