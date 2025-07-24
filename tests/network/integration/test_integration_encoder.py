# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from unittest import TestCase, skipIf

import torch

from deepparse.network import Encoder


class EncoderCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_torch_device = torch.device("cuda:0")
        cls.a_cpu_device = torch.device("cpu")

        cls.input_size_dim = 300
        cls.hidden_size = 1024
        cls.num_layers = 1
        cls.a_batch_size = 2
        cls.sequence_len = 6

    def setUp_encoder(self, device: torch.device) -> None:
        self.encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)
        self.encoder.to(device)  # we mount it into the device

        self.to_predict_tensor = torch.rand((self.a_batch_size, self.sequence_len, self.input_size_dim))
        self.to_predict_tensor = self.to_predict_tensor.to(device)

        self.a_lengths_list = [6, 4]

        self.a_longest_sequence_length = self.a_lengths_list[0]

    def assert_output_is_valid_dim(self, actual_predictions):
        self.assertEqual(self.a_batch_size, len(actual_predictions))
        for actual_prediction in actual_predictions:
            self.assertEqual(self.a_longest_sequence_length, actual_prediction.shape[0])
            self.assertEqual(self.hidden_size, actual_prediction.shape[1])


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class EncoderGPUTest(EncoderCase):
    def test_whenForwardStepGPU_thenStepIsOk(self):
        self.setUp_encoder(self.a_torch_device)
        predictions, _ = self.encoder.forward(self.to_predict_tensor, self.a_lengths_list)

        self.assert_output_is_valid_dim(predictions)


class EncoderCPUTest(EncoderCase):
    def test_whenForwardStepCPU_thenStepIsOk(self):
        self.setUp_encoder(self.a_cpu_device)

        predictions, _ = self.encoder.forward(self.to_predict_tensor, self.a_lengths_list)

        self.assert_output_is_valid_dim(predictions)


if __name__ == "__main__":
    unittest.main()
