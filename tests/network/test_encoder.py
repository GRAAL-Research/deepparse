from unittest import TestCase

import torch

from deepparse.network import Encoder


class EncoderTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_torch_device = torch.device("cpu")

        cls.input_size_dim = 300
        cls.hidden_size = 1024
        cls.num_layers = 1

    def setUp(self) -> None:
        self.encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)

    def test_whenInstantiateEncoder_thenParametersAreOk(self):
        self.assertEqual(self.input_size_dim, self.encoder.lstm.input_size)
        self.assertEqual(self.hidden_size, self.encoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, self.encoder.lstm.num_layers)
        self.assertEqual(self.a_torch_device, self.encoder.lstm.all_weights[0][0].device)
