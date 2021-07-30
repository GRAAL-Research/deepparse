# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import os
import pickle
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import torch

from deepparse import download_from_url
from deepparse.network import Decoder


class DecoderCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = TemporaryDirectory()
        cls.weights_dir = os.path.join(cls.temp_dir_obj.name, "./weights")

        download_from_url(file_name="decoder_hidden", saving_dir=cls.weights_dir, file_extension="p")

        cls.a_torch_device = torch.device("cuda:0")
        cls.a_cpu_device = torch.device("cpu")

        cls.input_size_dim = 1
        cls.hidden_size = 1024
        cls.num_layers = 1
        cls.a_batch_size = 2

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def setUp_encoder_decoder(self, output_size: int, device: torch.device) -> None:
        self.encoder = Decoder(self.input_size_dim, self.hidden_size, self.num_layers, output_size)
        self.encoder.to(device)  # we mount it into the device
        self.decoder_input_setUp(device)

    def decoder_input_setUp(self, device: torch.device):
        self.decoder_input = torch.tensor([[[-1.], [-1.]]], device=device)

        with open(os.path.join(self.weights_dir, "decoder_hidden.p"), "rb") as file:
            self.decoder_hidden_tensor = pickle.load(file)
        self.decoder_hidden_tensor = (self.decoder_hidden_tensor[0].to(device),
                                      self.decoder_hidden_tensor[1].to(device))

    def assert_predictions_is_valid_dim(self, actual_predictions, output_size: int):
        self.assertEqual(self.a_batch_size, actual_predictions.shape[0])
        self.assertEqual(output_size, actual_predictions.shape[1])

    def assert_hidden_is_valid_dim(self, actual_predictions):
        for actual_prediction in actual_predictions:
            self.assertEqual(self.num_layers, actual_prediction.shape[0])
            self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
            self.assertEqual(self.hidden_size, actual_prediction.shape[2])


@skipIf(not torch.cuda.is_available(), "no gpu available")
class DecoderGPUTest(DecoderCase):

    def test_whenForwardStepGPU_thenStepIsOk(self):
        output_size = 9
        self.setUp_encoder_decoder(output_size, self.a_torch_device)
        predictions, hidden = self.encoder.forward(self.decoder_input, self.decoder_hidden_tensor)

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)

    def test_whenForwardStepDim10GPU_thenStepIsOk(self):
        output_size = 10
        self.setUp_encoder_decoder(output_size, self.a_torch_device)
        predictions, hidden = self.encoder.forward(self.decoder_input, self.decoder_hidden_tensor)

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)


class DecoderCPUTest(DecoderCase):

    def test_whenForwardStepCPU_thenStepIsOk(self):
        output_size = 9
        self.setUp_encoder_decoder(output_size, self.a_cpu_device)
        predictions, hidden = self.encoder.forward(self.decoder_input, self.decoder_hidden_tensor)

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)

    def test_whenForwardStepDim10CPU_thenStepIsOk(self):
        output_size = 10
        self.setUp_encoder_decoder(output_size, self.a_cpu_device)
        predictions, hidden = self.encoder.forward(self.decoder_input, self.decoder_hidden_tensor)

        self.assert_predictions_is_valid_dim(predictions, output_size)
        self.assert_hidden_is_valid_dim(hidden)


if __name__ == "__main__":
    unittest.main()
