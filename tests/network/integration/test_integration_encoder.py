# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import os
import pickle
import shutil
import unittest
from unittest import TestCase, skipIf

import torch

from deepparse import download_from_url
from deepparse.network import Encoder


@skipIf(not torch.cuda.is_available(), "no gpu available")
class EncoderTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_torch_device = torch.device("cuda:0")

        cls.input_size_dim = 300
        cls.hidden_size = 1024
        cls.num_layers = 1

        cls.weights_dir = "./weights"
        download_from_url(file_name="to_predict_fasttext", saving_dir=cls.weights_dir, file_extension=".p")

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.weights_dir):
            shutil.rmtree(cls.weights_dir)

    def setUp(self) -> None:
        self.encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)
        self.encoder.to(self.a_torch_device)  # we mount it into the device
        self.encoder_input_setUp()

    def encoder_input_setUp(self):
        # we use the fasttext case since easier
        with open(os.path.join(self.weights_dir, "to_predict_fasttext.p"), "rb") as file:
            self.to_predict_tensor = pickle.load(file)
        self.to_predict_tensor = self.to_predict_tensor.to(self.a_torch_device)

        self.a_lengths_tensor = torch.tensor([6, 6], device=self.a_torch_device)
        self.a_batch_size = 2

        self.max_length = self.a_lengths_tensor[0].item()

    def assert_output_is_valid_dim(self, actual_predictions):
        for actual_prediction in actual_predictions:
            self.assertEqual(self.num_layers, actual_prediction.shape[0])
            self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
            self.assertEqual(self.hidden_size, actual_prediction.shape[2])

    def test_whenForwardStep_thenStepIsOk(self):
        predictions = self.encoder.forward(self.to_predict_tensor, self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)


if __name__ == "__main__":
    unittest.main()
