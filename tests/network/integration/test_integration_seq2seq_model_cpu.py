# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access
import os
import unittest
from unittest import skipIf
from unittest.mock import patch

import torch

from deepparse.network import Seq2SeqModel
from ..integration.base import Seq2SeqIntegrationTestCase


@skipIf(not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner")
class Seq2SeqIntegrationTest(Seq2SeqIntegrationTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.pre_trained_seq2seq_model = Seq2SeqModel(self.a_cpu_device,
                                                      input_size=self.input_size,
                                                      encoder_hidden_size=self.encoder_hidden_size,
                                                      encoder_num_layers=self.num_layers,
                                                      decoder_hidden_size=self.decoder_hidden_size,
                                                      decoder_num_layers=self.num_layers,
                                                      output_size=self.output_size)
        self.encoder_input_setUp("fasttext",
                                 self.a_cpu_device)  # fasttext since the simplest case (bpemb use a embedding layer)
        self.none_target = None  # No target (for teacher forcing)
        self.a_value_greater_than_threshold = 0.1

        self.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=self.a_cpu_device)

    def test_whenEncoderStep_thenEncoderStepIsOk(self):
        # encoding for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"

        decoder_input, decoder_hidden, encoder_outputs = self.pre_trained_seq2seq_model._encoder_step(
            self.to_predict_tensor, self.a_lengths_tensor, self.a_batch_size)

        self.assertEqual(decoder_input.shape[1], self.a_batch_size)
        self.assertTrue(decoder_input[0][0] == self.begin_of_sequence_idx)
        self.assertTrue(decoder_input[0][1] == self.begin_of_sequence_idx)

        self.assertEqual(encoder_outputs.shape[0], self.a_batch_size)
        self.assertTrue(encoder_outputs.shape[1] == self.a_target_vector.shape[1])  # number of tokens (padded)
        self.assertTrue(encoder_outputs.shape[2] == self.encoder_hidden_size)

        self.assertEqual(len(decoder_hidden), self.a_batch_size)
        self.assertEqual(decoder_hidden[0].shape[2], self.encoder_hidden_size)
        self.assertEqual(decoder_hidden[0].shape[2], self.encoder_hidden_size)
        self.assertEqual(decoder_hidden[1].shape[2], self.encoder_hidden_size)
        self.assertEqual(decoder_hidden[1].shape[2], self.encoder_hidden_size)

    def test_whenDecoderStep_thenDecoderStepIsOk(self):
        # decoding for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.encoder_output_setUp(self.a_cpu_device)
        self.decoder_input_setUp()

        actual_prediction_sequence = self.pre_trained_seq2seq_model._decoder_step(self.decoder_input,
                                                                                  self.decoder_hidden_tensor,
                                                                                  self.encoder_hidden, self.none_target,
                                                                                  self.a_lengths_tensor,
                                                                                  self.a_batch_size)

        self.assert_output_is_valid_dim(actual_prediction_sequence, output_dim=self.number_of_tags)

    def test_whenDecoderStepTeacherForcing_thenDecoderStepIsOk(self):
        # decoding for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.encoder_output_setUp(self.a_cpu_device)
        self.decoder_input_setUp()

        actual_prediction_sequence = self.pre_trained_seq2seq_model._decoder_step(
            self.decoder_input, self.decoder_hidden_tensor, self.encoder_hidden, self.a_target_vector,
            self.a_lengths_tensor, self.a_batch_size)

        self.assert_output_is_valid_dim(actual_prediction_sequence, output_dim=self.number_of_tags)

    @patch("deepparse.network.seq2seq.random.random")
    def test_whenDecoderStepWithTarget_thenUsesTarget(self, random_mock):
        random_mock.return_value = self.a_value_greater_than_threshold

        # decoding for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.encoder_output_setUp(self.a_cpu_device)
        self.decoder_input_setUp()

        self.pre_trained_seq2seq_model._decoder_step(self.decoder_input, self.decoder_hidden_tensor,
                                                     self.encoder_hidden, self.a_target_vector, self.a_lengths_tensor,
                                                     self.a_batch_size)

        random_mock.assert_called_once()

    @patch("deepparse.network.seq2seq.random.random")
    def test_whenDecoderStepWithoutTarget_thenDontUsesTarget(self, random_mock):
        random_mock.return_value = self.a_value_greater_than_threshold

        # decoding for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.encoder_output_setUp(self.a_cpu_device)
        self.decoder_input_setUp()

        self.pre_trained_seq2seq_model._decoder_step(self.decoder_input, self.decoder_hidden_tensor,
                                                     self.encoder_hidden, self.none_target, self.a_lengths_tensor,
                                                     self.a_batch_size)

        random_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
