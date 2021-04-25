import unittest
from unittest import TestCase

from deepparse.network import Encoder


class EncoderTest(TestCase):

    def setUp(self) -> None:
        self.input_size_dim = 300
        self.hidden_size = 1024
        self.num_layers = 1

        self.encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)

    def test_whenInstantiateEncoder_thenParametersAreOk(self):
        self.assertEqual(self.input_size_dim, self.encoder.lstm.input_size)
        self.assertEqual(self.hidden_size, self.encoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, self.encoder.lstm.num_layers)


if __name__ == "__main__":
    unittest.main()
