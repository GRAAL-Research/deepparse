# Pylint raise error for the getitem method mocking
# pylint: disable=unnecessary-dunder-call

import unittest
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import torch

from deepparse.network import Decoder


class DecoderTest(TestCase):
    def setUp(self) -> None:
        self.input_size_dim = 1
        self.hidden_size = 1024
        self.num_layers = 1
        self.output_size = 9
        # A batch of 3 sequence of length (respectively): 2, 4 and 3. Thus, the longest one is 4.
        self.a_lengths_list = [1, 4, 3]

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        decoder = Decoder(
            self.input_size_dim,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            attention_mechanism=False,
        )

        self.assertEqual(self.input_size_dim, decoder.lstm.input_size)
        self.assertEqual(self.hidden_size, decoder.lstm.hidden_size)
        self.assertEqual(self.num_layers, decoder.lstm.num_layers)
        self.assertEqual(self.output_size, decoder.linear.out_features)

    def test_whenInstantiateASeq2SeqAttModel_thenParametersAreOk(self):
        decoder = Decoder(
            self.input_size_dim,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            attention_mechanism=True,
        )

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
                    decoder = Decoder(
                        self.input_size_dim,
                        self.hidden_size,
                        self.num_layers,
                        self.output_size,
                        attention_mechanism=False,
                    )
                    to_predict_mock = MagicMock()
                    hidden_mock = MagicMock()
                    encoder_mock = MagicMock()

                    decoder.forward(to_predict_mock, hidden_mock, encoder_mock, self.a_lengths_list)

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
                                decoder = Decoder(
                                    self.input_size_dim,
                                    self.hidden_size,
                                    self.num_layers,
                                    self.output_size,
                                    attention_mechanism=True,
                                )
                                to_predict_mock = MagicMock()
                                hidden_mock = MagicMock()
                                encoder_mock = MagicMock()
                                _, _, attention_weights = decoder.forward(
                                    to_predict_mock,
                                    hidden_mock,
                                    encoder_mock,
                                    self.a_lengths_list,
                                )
                                self.assertIsNotNone(attention_weights)

    def test_givenAttForwardWithRealTensors_thenMaskZeroesAttentionBeyondEachLength(self):
        # Real tensors (no mocking of matmul) to exercise the actual masking. The mask must run on the model's
        # device and zero out attention weights for positions past each sequence's length.
        decoder = Decoder(
            self.input_size_dim,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            attention_mechanism=True,
        )

        batch_size = 2
        sequence_length = 3
        encoder_outputs = torch.ones(batch_size, sequence_length, self.hidden_size)
        hidden = (
            torch.ones(self.num_layers, batch_size, self.hidden_size),
            torch.ones(self.num_layers, batch_size, self.hidden_size),
        )
        to_predict = torch.ones(1, batch_size, self.input_size_dim)
        lengths = [3, 1]  # The second sequence has length 1: positions 1 and 2 must be masked out.

        _, _, attention_weights = decoder.forward(to_predict, hidden, encoder_outputs, lengths)

        # Shape is (batch, 1, sequence_length).
        self.assertEqual(attention_weights[1, 0, 1].item(), 0.0)
        self.assertEqual(attention_weights[1, 0, 2].item(), 0.0)
        # The full-length sequence keeps non-zero attention on every position, and weights sum to 1.
        self.assertGreater(attention_weights[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(attention_weights[0, 0].sum().item(), 1.0, places=5)
        self.assertAlmostEqual(attention_weights[1, 0].sum().item(), 1.0, places=5)

    def test_whenDecoderNotAttForward_thenReturnAttWeightsToNone(self):
        with patch("deepparse.network.nn.LSTM") as lstm_mock:
            output_mock = MagicMock()
            hidden_mock = MagicMock()
            lstm_mock().return_value = output_mock, hidden_mock

            with patch("deepparse.network.nn.Linear") as linear_mock:
                linear_output = MagicMock()
                linear_mock().return_value = linear_output
                with patch("deepparse.network.nn.LogSoftmax"):
                    decoder = Decoder(
                        self.input_size_dim,
                        self.hidden_size,
                        self.num_layers,
                        self.output_size,
                        attention_mechanism=False,
                    )
                    to_predict_mock = MagicMock()
                    hidden_mock = MagicMock()
                    encoder_mock = MagicMock()

                    _, _, attention_weights = decoder.forward(
                        to_predict_mock, hidden_mock, encoder_mock, self.a_lengths_list
                    )
                    self.assertIsNone(attention_weights)


if __name__ == "__main__":
    unittest.main()
