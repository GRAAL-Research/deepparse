import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock, call

import torch

from deepparse.network import Decoder


class DecoderTest(TestCase):

    def setUp(self) -> None:
        self.input_size_dim = 1
        self.hidden_size = 1024
        self.num_layers = 1
        self.output_size = 9

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        decoder = Decoder(self.input_size_dim,
                          self.hidden_size,
                          self.num_layers,
                          self.output_size,
                          attention_mechanism=False)

        self.assertEqual(self.input_size_dim, decoder.lstm.input_size)
        self.assertEqual(self.hidden_size, decoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, decoder.lstm.num_layers)
        self.assertEqual(self.output_size, decoder.linear.out_features)

    def test_whenInstantiateASeq2SeqAttModel_thenParametersAreOk(self):
        decoder = Decoder(self.input_size_dim,
                          self.hidden_size,
                          self.num_layers,
                          self.output_size,
                          attention_mechanism=True)

        self.assertEqual(self.input_size_dim + self.hidden_size, decoder.lstm.input_size)
        self.assertEqual(self.hidden_size, decoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, decoder.lstm.num_layers)
        self.assertEqual(self.output_size, decoder.linear.out_features)

    def test_whenDecoderForward_thenPass(self):
        with patch("deepparse.network.nn.LSTM") as lstm_mock:
            output_mock = MagicMock()
            hidden_mock = MagicMock()
            lstm_mock().return_value = output_mock, hidden_mock

            with patch("deepparse.network.nn.Linear") as linear_mock:
                linear_output = MagicMock()
                linear_mock().return_value = linear_output
                with patch("deepparse.network.nn.LogSoftmax") as log_soft_max_mock:
                    decoder = Decoder(self.input_size_dim,
                                      self.hidden_size,
                                      self.num_layers,
                                      self.output_size,
                                      attention_mechanism=False)
                    to_predict_mock = MagicMock()
                    hidden_mock = MagicMock()
                    encoder_mock = MagicMock()
                    lengths_mock = MagicMock()
                    decoder.forward(to_predict_mock, hidden_mock, encoder_mock, lengths_mock)

                    lstm_mock.assert_has_calls([call()(to_predict_mock.float(), hidden_mock)])
                    linear_mock.assert_has_calls([call()(output_mock.__getitem__())])
                    log_soft_max_mock.assert_has_calls([call()(linear_output)])

    def test_whenDecoderAttForward_thenReturnAttWeights(self):
        with patch("deepparse.network.nn.LSTM") as lstm_mock:
            output_mock = MagicMock()
            hidden_mock = MagicMock()
            lstm_mock().return_value = output_mock, hidden_mock

            with patch("deepparse.network.nn.Linear") as linear_mock:
                linear_mock().return_value = MagicMock()
                with patch("deepparse.network.torch.tanh") as tanh_mock:
                    tanh_mock().return_value = MagicMock()
                    with patch("deepparse.network.torch.matmul") as matmul_mock:
                        matmul_mock().return_value = MagicMock()
                        with patch("deepparse.network.torch.cat") as cat_mock:
                            cat_mock().return_value = MagicMock()
                            with patch("deepparse.network.nn.LogSoftmax"):
                                decoder = Decoder(self.input_size_dim,
                                                  self.hidden_size,
                                                  self.num_layers,
                                                  self.output_size,
                                                  attention_mechanism=True)
                                to_predict_mock = MagicMock()
                                hidden_mock = MagicMock()
                                encoder_mock = MagicMock()
                                lengths_mock = torch.ones(1, 2)
                                _, _, attention_weights = decoder.forward(to_predict_mock, hidden_mock, encoder_mock,
                                                                          lengths_mock)
                                self.assertIsNotNone(attention_weights)

    def test_whenDecoderNotAttForward_thenReturnAttWeightsToNone(self):
        with patch("deepparse.network.nn.LSTM") as lstm_mock:
            output_mock = MagicMock()
            hidden_mock = MagicMock()
            lstm_mock().return_value = output_mock, hidden_mock

            with patch("deepparse.network.nn.Linear") as linear_mock:
                linear_output = MagicMock()
                linear_mock().return_value = linear_output
                with patch("deepparse.network.nn.LogSoftmax"):
                    decoder = Decoder(self.input_size_dim,
                                      self.hidden_size,
                                      self.num_layers,
                                      self.output_size,
                                      attention_mechanism=False)
                    to_predict_mock = MagicMock()
                    hidden_mock = MagicMock()
                    encoder_mock = MagicMock()
                    lengths_mock = MagicMock()
                    _, _, attention_weights = decoder.forward(to_predict_mock, hidden_mock, encoder_mock, lengths_mock)
                    self.assertIsNone(attention_weights)


if __name__ == "__main__":
    unittest.main()
