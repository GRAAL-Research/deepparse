import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock, call

from deepparse.network import Encoder


class EncoderTest(TestCase):

    def setUp(self) -> None:
        self.input_size_dim = 300
        self.hidden_size = 1024
        self.num_layers = 1

    def test_whenInstantiateEncoder_thenParametersAreOk(self):
        encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)
        self.assertEqual(self.input_size_dim, encoder.lstm.input_size)
        self.assertEqual(self.hidden_size, encoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, encoder.lstm.num_layers)

    def test_whenEncoderForward_thenPass(self):
        with patch("deepparse.network.nn.LSTM") as lstm_mock:
            output_mock = MagicMock()
            hidden_mock = MagicMock()
            lstm_mock().return_value = output_mock, hidden_mock

            with patch("deepparse.network.encoder.pack_padded_sequence") as pack_padded_sequence_mock:
                packed_sequence_mock = MagicMock()
                pack_padded_sequence_mock.return_value = packed_sequence_mock

                encoder = Encoder(self.input_size_dim, self.hidden_size, self.num_layers)
                to_predict_mock = MagicMock()
                lengths_tensor_mock = MagicMock()
                encoder.forward(to_predict_mock, lengths_tensor_mock)

                pack_padded_sequence_mock.assert_has_calls(
                    [call(to_predict_mock, lengths_tensor_mock.cpu(), batch_first=True, enforce_sorted=False)])
                lstm_mock.assert_has_calls([call()(packed_sequence_mock)])


if __name__ == "__main__":
    unittest.main()
