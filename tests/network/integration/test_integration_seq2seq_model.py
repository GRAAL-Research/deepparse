# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access
import unittest
from unittest import skipIf
from unittest.mock import patch

import torch

from deepparse.network import Seq2SeqModel
from ..integration.base import Seq2SeqIntegrationTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class Seq2SeqIntegrationTest(Seq2SeqIntegrationTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.pre_trained_seq2seq_model = Seq2SeqModel(self.a_torch_device)
        self.encoder_input_setUp("fasttext")  # fasttext since the simplest case (bpemb use a embedding layer)
        self.none_target = None  # No target (for teacher forcing)
        self.a_value_greater_than_threshold = 0.1

    def test_whenEncoderStep_thenEncoderStepIsOk(self):
        # encoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'

        actual_decoder_input, actual_decoder_hidden = self.pre_trained_seq2seq_model._encoder_step(
            self.to_predict_tensor, self.a_lengths_tensor, self.a_batch_size)

        self.assertEqual(self.a_batch_size, actual_decoder_input.shape[1])
        self.assertTrue(actual_decoder_input[0][0] == self.begin_of_sequence_idx)
        self.assertTrue(actual_decoder_input[0][1] == self.begin_of_sequence_idx)

        self.assertEqual(self.a_batch_size, len(actual_decoder_hidden))
        self.assertEqual(self.encoder_hidden_size, actual_decoder_hidden[0].shape[2])
        self.assertEqual(self.encoder_hidden_size, actual_decoder_hidden[1].shape[2])

    def test_whenDecoderStep_thenDecoderStepIsOk(self):
        # decoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.encoder_output_setUp()
        self.decoder_input_setUp()

        actual_prediction_sequence = self.pre_trained_seq2seq_model._decoder_step(self.decoder_input,
                                                                                  self.decoder_hidden_tensor,
                                                                                  self.none_target, self.max_length,
                                                                                  self.a_batch_size)

        self.assert_output_is_valid_dim(actual_prediction_sequence)

    def test_whenDecoderStepTeacherForcing_thenDecoderStepIsOk(self):
        # decoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.encoder_output_setUp()
        self.decoder_input_setUp()

        actual_prediction_sequence = self.pre_trained_seq2seq_model._decoder_step(self.decoder_input,
                                                                                  self.decoder_hidden_tensor,
                                                                                  self.a_target_vector, self.max_length,
                                                                                  self.a_batch_size)

        self.assert_output_is_valid_dim(actual_prediction_sequence)

    @patch("deepparse.network.seq2seq.random.random")
    def test_whenDecoderStepWithTarget_thenUsesTarget(self, random_mock):
        random_mock.return_value = self.a_value_greater_than_threshold

        # decoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.encoder_output_setUp()
        self.decoder_input_setUp()

        _ = self.pre_trained_seq2seq_model._decoder_step(self.decoder_input, self.decoder_hidden_tensor,
                                                         self.a_target_vector, self.max_length, self.a_batch_size)

        random_mock.assert_called_once()

    @patch("deepparse.network.seq2seq.random.random")
    def test_whenDecoderStepWithoutTarget_thenDontUsesTarget(self, random_mock):
        random_mock.return_value = self.a_value_greater_than_threshold

        # decoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.encoder_output_setUp()
        self.decoder_input_setUp()

        _ = self.pre_trained_seq2seq_model._decoder_step(self.decoder_input, self.decoder_hidden_tensor,
                                                         self.none_target, self.max_length, self.a_batch_size)

        random_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
