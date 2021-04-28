# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the _load_weights
# pylint: disable=protected-access, unused-argument
import unittest
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

from deepparse.network import Seq2SeqModel


class Seq2SeqTest(TestCase):

    def setUp(self) -> None:
        self.a_torch_device = torch.device("cpu")

        self.encoder_input_size_dim = 300
        self.encoder_hidden_size = 1024
        self.encoder_num_layers = 1
        self.decoder_input_size_dim = 1
        self.decoder_hidden_size = 1024
        self.decoder_num_layers = 1
        self.decoder_output_size = 9

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        seq2seq_model = Seq2SeqModel(self.a_torch_device, output_size=self.decoder_output_size)
        self.assertEqual(self.a_torch_device, seq2seq_model.device)

        self.assertEqual(self.encoder_input_size_dim, seq2seq_model.encoder.lstm.input_size)
        self.assertEqual(self.encoder_hidden_size, seq2seq_model.encoder.lstm.hidden_size)
        self.assertEqual(self.encoder_num_layers, seq2seq_model.encoder.lstm.num_layers)
        self.assertEqual(self.a_torch_device, seq2seq_model.encoder.lstm.all_weights[0][0].device)

        self.assertEqual(self.decoder_input_size_dim, seq2seq_model.decoder.lstm.input_size)
        self.assertEqual(self.decoder_hidden_size, seq2seq_model.decoder.lstm.hidden_size)
        self.assertEqual(self.decoder_num_layers, seq2seq_model.decoder.lstm.num_layers)
        self.assertEqual(self.decoder_output_size, seq2seq_model.decoder.linear.out_features)
        self.assertEqual(self.a_torch_device, seq2seq_model.decoder.lstm.all_weights[0][0].device)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    def test_givenSeq2seqModel_whenLoadPreTrainedWeightsVerbose_thenWarningsRaised(self, torch_nn_mock, torch_mock,
                                                                                   isfile_mock, last_version_mock):
        seq2seq_model = Seq2SeqModel(self.a_torch_device, verbose=True, output_size=self.decoder_output_size)
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights"):
            with self.assertWarns(UserWarning):
                seq2seq_model._load_pre_trained_weights("a_model_type")

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    def test_givenSeq2seqModel_whenLoadPreTrainedWeightsNotVerbose_thenWarningsNotRaised(
            self, torch_nn_mock, torch_mock, isfile_mock, last_version_mock):
        seq2seq_model = Seq2SeqModel(self.a_torch_device, verbose=False, output_size=self.decoder_output_size)
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights"):
            with pytest.warns(None) as record:
                seq2seq_model._load_pre_trained_weights("a_model_type")
            self.assertEqual(0, len(record))


if __name__ == "__main__":
    unittest.main()
