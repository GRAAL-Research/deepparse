# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from unittest import TestCase, skipIf

import torch

from deepparse.network import Decoder


class DecoderCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_torch_device = torch.device("cuda:0")
        cls.a_cpu_device = torch.device("cpu")

        cls.input_size_dim = 1
        cls.hidden_size = 1024
        cls.num_layers = 1
        cls.a_batch_size = 2
        cls.sequence_len = 1

    def setUp_encoder_decoder(self, output_size: int, device: torch.device, attention_mechanism) -> None:
        self.decoder = Decoder(
            self.input_size_dim,
            self.hidden_size,
            self.num_layers,
            output_size,
            attention_mechanism=attention_mechanism,
        )
        self.decoder.to(device)  # we mount it into the device
        self.decoder_input_setUp(device)

    def decoder_input_setUp(self, device: torch.device):
        self.decoder_input = torch.tensor([[[-1], [-1]]], device=device)
        self.a_lengths_list = [self.sequence_len, self.sequence_len]

        self.decoder_hidden_tensor = (
            torch.rand((self.num_layers, self.a_batch_size, self.hidden_size)).to(device),
            torch.rand((self.num_layers, self.a_batch_size, self.hidden_size)).to(device),
        )
        self.decoder_output = torch.rand((1, self.sequence_len, self.hidden_size), device=device)

    def assert_predictions_is_valid_dim(self, actual_predictions, output_size: int):
        self.assertEqual(self.a_batch_size, actual_predictions.shape[0])
        self.assertEqual(output_size, actual_predictions.shape[1])

    def assert_hidden_is_valid_dim(self, actual_predictions):
        for actual_prediction in actual_predictions:
            self.assertEqual(self.num_layers, actual_prediction.shape[0])
            self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
            self.assertEqual(self.hidden_size, actual_prediction.shape[2])


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class DecoderGPUTest(DecoderCase):
    def test_whenForwardStepGPU_thenStepIsOk(self):
        output_size = 9
        self.setUp_encoder_decoder(output_size, self.a_torch_device, attention_mechanism=False)
        predictions, hidden, att_weights = self.decoder.forward(
            self.decoder_input,
            self.decoder_hidden_tensor,
            self.decoder_output,
            self.a_lengths_list,
        )

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)
        self.assertIsNone(att_weights)

    def test_whenForwardAttStepGPU_thenStepIsOk(self):
        output_size = 9
        self.setUp_encoder_decoder(output_size, self.a_torch_device, attention_mechanism=True)
        predictions, hidden, att_weights = self.decoder.forward(
            self.decoder_input,
            self.decoder_hidden_tensor,
            self.decoder_output,
            self.a_lengths_list,
        )

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)
        self.assertIsNotNone(att_weights)

    def test_whenForwardStepDim10GPU_thenStepIsOk(self):
        output_size = 10
        self.setUp_encoder_decoder(output_size, self.a_torch_device, attention_mechanism=False)
        predictions, hidden, att_weights = self.decoder.forward(
            self.decoder_input,
            self.decoder_hidden_tensor,
            self.decoder_output,
            self.a_lengths_list,
        )

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)
        self.assertIsNone(att_weights)


class DecoderCPUTest(DecoderCase):
    def test_whenForwardStepCPU_thenStepIsOk(self):
        output_size = 9
        self.setUp_encoder_decoder(output_size, self.a_cpu_device, attention_mechanism=False)
        predictions, hidden, att_weights = self.decoder.forward(
            self.decoder_input,
            self.decoder_hidden_tensor,
            self.decoder_output,
            self.a_lengths_list,
        )

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)
        self.assertIsNone(att_weights)

    def test_whenForwardAttStepCPU_thenStepIsOk(self):
        output_size = 9
        self.setUp_encoder_decoder(output_size, self.a_cpu_device, attention_mechanism=True)
        predictions, hidden, att_weights = self.decoder.forward(
            self.decoder_input,
            self.decoder_hidden_tensor,
            self.decoder_output,
            self.a_lengths_list,
        )

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)
        self.assertIsNotNone(att_weights)

    def test_whenForwardStepDim10CPU_thenStepIsOk(self):
        output_size = 10
        self.setUp_encoder_decoder(output_size, self.a_cpu_device, attention_mechanism=False)
        predictions, hidden, att_weights = self.decoder.forward(
            self.decoder_input,
            self.decoder_hidden_tensor,
            self.decoder_output,
            self.a_lengths_list,
        )

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)
        self.assertIsNone(att_weights)


if __name__ == "__main__":
    unittest.main()
