# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import pickle
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import torch

from deepparse import download_from_public_repository
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

        cls.temp_dir_obj = TemporaryDirectory()
        cls.weights_dir = os.path.join(cls.temp_dir_obj.name, "weights")
        download_from_public_repository(
            file_name="to_predict_fasttext",
            saving_dir=cls.weights_dir,
            file_extension="p",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def setUp_encoder(self, device: torch.device) -> None:
        self.encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)
        self.encoder.to(device)  # we mount it into the device

        with open(os.path.join(self.weights_dir, "to_predict_fasttext.p"), "rb") as file:
            self.to_predict_tensor = pickle.load(file)
        self.to_predict_tensor = self.to_predict_tensor.to(device)

        self.a_lengths_tensor = torch.tensor([6, 6], device=device)

        self.max_length = self.a_lengths_tensor[0].item()

    def assert_output_is_valid_dim(self, actual_predictions):
        self.assertEqual(self.a_batch_size, len(actual_predictions))
        for actual_prediction in actual_predictions:
            self.assertEqual(self.max_length, actual_prediction.shape[0])
            self.assertEqual(self.hidden_size, actual_prediction.shape[1])


@skipIf(not torch.cuda.is_available(), "no gpu available")
class EncoderGPUTest(EncoderCase):
    def test_whenForwardStepGPU_thenStepIsOk(self):
        self.setUp_encoder(self.a_torch_device)
        predictions, _ = self.encoder.forward(self.to_predict_tensor, self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)


class EncoderCPUTest(EncoderCase):
    def test_whenForwardStepCPU_thenStepIsOk(self):
        self.setUp_encoder(self.a_cpu_device)

        predictions, _ = self.encoder.forward(self.to_predict_tensor, self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)


if __name__ == "__main__":
    unittest.main()
