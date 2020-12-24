# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

import pickle
import unittest
from unittest import TestCase, skipIf

import torch

from deepparse.network import Decoder


@skipIf(not torch.cuda.is_available(), "no gpu available")
class DecoderTest(TestCase):

    def setUp(self) -> None:
        self.a_torch_device = torch.device("cuda:0")

        self.input_size_dim = 1
        self.hidden_size = 1024
        self.num_layers = 1
        self.output_size = 9

        self.a_batch_size = 2

        self.encoder = Decoder(self.input_size_dim, self.hidden_size, self.num_layers, self.output_size)
        self.encoder.to(self.a_torch_device)  # we mount it into the device
        self.decoder_input_setUp()

    def decoder_input_setUp(self):
        self.decoder_input = torch.tensor([[[-1.], [-1.]]], device=self.a_torch_device)

        file = open("./tests/network/integration/decoder_hidden.p", "rb")
        self.decoder_hidden_tensor = pickle.load(file)
        self.decoder_hidden_tensor = (self.decoder_hidden_tensor[0].to(self.a_torch_device),
                                      self.decoder_hidden_tensor[1].to(self.a_torch_device))
        file.close()

    def assert_predictions_is_valid_dim(self, actual_predictions):
        self.assertEqual(self.a_batch_size, actual_predictions.shape[0])
        self.assertEqual(self.output_size, actual_predictions.shape[1])

    def assert_hidden_is_valid_dim(self, actual_predictions):
        for actual_prediction in actual_predictions:
            self.assertEqual(self.num_layers, actual_prediction.shape[0])
            self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
            self.assertEqual(self.hidden_size, actual_prediction.shape[2])

    def test_whenForwardStep_thenStepIsOk(self):
        predictions, hidden = self.encoder.forward(self.decoder_input, self.decoder_hidden_tensor)

        self.assert_predictions_is_valid_dim(predictions)
        self.assert_hidden_is_valid_dim(hidden)


if __name__ == "__main__":
    unittest.main()
