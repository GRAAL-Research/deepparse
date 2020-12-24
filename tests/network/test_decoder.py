import unittest
from unittest import TestCase

import torch

from deepparse.network import Decoder


class DecoderTest(TestCase):

    def setUp(self) -> None:
        self.a_torch_device = torch.device("cpu")

        self.input_size_dim = 1
        self.hidden_size = 1024
        self.num_layers = 1
        self.output_size = 9

        self.decoder = Decoder(self.input_size_dim, self.hidden_size, self.num_layers, self.output_size)

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        self.assertEqual(self.input_size_dim, self.decoder.lstm.input_size)
        self.assertEqual(self.hidden_size, self.decoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, self.decoder.lstm.num_layers)
        self.assertEqual(self.output_size, self.decoder.linear.out_features)
        self.assertEqual(self.a_torch_device, self.decoder.lstm.all_weights[0][0].device)


if __name__ == "__main__":
    unittest.main()
