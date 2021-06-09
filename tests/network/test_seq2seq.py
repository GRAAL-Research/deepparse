# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the _load_weights
# pylint: disable=protected-access, unused-argument, not-callable

import unittest
from unittest import TestCase
from unittest import skipIf
from unittest.mock import patch, MagicMock, call

import pytest
import torch

from deepparse.network import Seq2SeqModel


class Seq2SeqTest(TestCase):

    def setUp(self) -> None:
        self.a_torch_device = torch.device("cuda:0")
        self.a_cpu_device = torch.device("cpu")

        self.encoder_input_size_dim = 300
        self.encoder_hidden_size = 1024
        self.encoder_num_layers = 1
        self.decoder_input_size_dim = 1
        self.decoder_hidden_size = 1024
        self.decoder_num_layers = 1
        self.decoder_output_size = 9

        self.a_fake_retrain_path = "a/fake/path/retrain/model"

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        seq2seq_model = Seq2SeqModel(self.a_cpu_device, output_size=self.decoder_output_size)

        self.assertEqual(self.a_cpu_device, seq2seq_model.device)

        self.assertEqual(self.encoder_input_size_dim, seq2seq_model.encoder.lstm.input_size)
        self.assertEqual(self.encoder_hidden_size, seq2seq_model.encoder.lstm.hidden_size)
        self.assertEqual(self.encoder_num_layers, seq2seq_model.encoder.lstm.num_layers)
        self.assertEqual(self.a_cpu_device, seq2seq_model.encoder.lstm.all_weights[0][0].device)

        self.assertEqual(self.decoder_input_size_dim, seq2seq_model.decoder.lstm.input_size)
        self.assertEqual(self.decoder_hidden_size, seq2seq_model.decoder.lstm.hidden_size)
        self.assertEqual(self.decoder_num_layers, seq2seq_model.decoder.lstm.num_layers)
        self.assertEqual(self.decoder_output_size, seq2seq_model.decoder.linear.out_features)
        self.assertEqual(self.a_cpu_device, seq2seq_model.decoder.lstm.all_weights[0][0].device)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_whenInstantiateASeq2SeqModelGPU_thenParametersAreOk(self):
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

    def test_whenSameOutput_thenReturnTrue(self):
        seq2seq_model = Seq2SeqModel(self.a_cpu_device, output_size=self.decoder_output_size)
        self.assertTrue(seq2seq_model.same_output_dim(self.decoder_output_size))

    def test_whenNotSameOutput_thenReturnFalse(self):
        seq2seq_model = Seq2SeqModel(self.a_cpu_device, output_size=self.decoder_output_size)
        self.assertFalse(seq2seq_model.same_output_dim(self.decoder_output_size - 1))

    def test_whenHandleNewOutputDim_thenProperlyHandleNewDim(self):
        seq2seq_model = Seq2SeqModel(self.a_cpu_device, output_size=self.decoder_output_size)

        a_new_dim = 1
        seq2seq_model.handle_new_output_dim(a_new_dim)

        expected = a_new_dim
        actual = seq2seq_model.output_size
        self.assertEqual(expected, actual)

        actual = seq2seq_model.decoder.linear.out_features
        self.assertEqual(expected, actual)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenSeq2seqModel_whenLoadPreTrainedWeightsVerboseGPU_thenWarningsRaised(
            self, torch_nn_mock, torch_mock, isfile_mock, last_version_mock):
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
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenSeq2seqModel_whenLoadPreTrainedWeightsNotVerboseGPU_thenWarningsNotRaised(
            self, torch_nn_mock, torch_mock, isfile_mock, last_version_mock):
        seq2seq_model = Seq2SeqModel(self.a_torch_device, verbose=False, output_size=self.decoder_output_size)
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights"):
            with pytest.warns(None) as record:
                seq2seq_model._load_pre_trained_weights("a_model_type")
            self.assertEqual(0, len(record))

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    def test_givenSeq2seqModel_whenLoadPreTrainedWeightsVerboseCPU_thenWarningsRaised(
            self, torch_nn_mock, torch_mock, isfile_mock, last_version_mock):
        seq2seq_model = Seq2SeqModel(self.a_cpu_device, verbose=True, output_size=self.decoder_output_size)
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights"):
            with self.assertWarns(UserWarning):
                seq2seq_model._load_pre_trained_weights("a_model_type")

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    def test_givenSeq2seqModel_whenLoadPreTrainedWeightsNotVerboseCPU_thenWarningsNotRaised(
            self, torch_nn_mock, torch_mock, isfile_mock, last_version_mock):
        seq2seq_model = Seq2SeqModel(self.a_cpu_device, verbose=False, output_size=self.decoder_output_size)
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights"):
            with pytest.warns(None) as record:
                seq2seq_model._load_pre_trained_weights("a_model_type")
            self.assertEqual(0, len(record))

    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    def test_givenSeq2SeqModelRetrained_whenLoadRetrainedWeights_thenLoadProperly(self, torch_nn_mock, torch_mock):
        all_layers_params_mock = MagicMock()
        all_layers_params_mock.__getitem__().__len__.return_value = self.decoder_output_size
        torch_mock.load.return_value = all_layers_params_mock

        seq2seq_model = Seq2SeqModel(self.a_cpu_device, verbose=True, output_size=self.decoder_output_size)
        seq2seq_model._load_weights(self.a_fake_retrain_path)

        torch_mock.assert_has_calls([call.load(self.a_fake_retrain_path, map_location=self.a_cpu_device)])

        torch_nn_mock.assert_called()
        torch_nn_mock.asser_has_calls([call(all_layers_params_mock)])

    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.torch.nn.Module.load_state_dict")
    def test_givenSeq2SeqModelRetrained_whenLoadRetrainedWeightsNewTagModel_thenLoadProperDict(
            self, torch_nn_mock, torch_mock):
        all_layers_params_mock = MagicMock(spec=dict)
        all_layers_params_mock.__getitem__().__len__.return_value = self.decoder_output_size
        torch_mock.load.return_value = all_layers_params_mock

        seq2seq_model = Seq2SeqModel(self.a_cpu_device, verbose=True, output_size=self.decoder_output_size)
        seq2seq_model._load_weights(self.a_fake_retrain_path)

        all_layers_params_mock.get.assert_called()


if __name__ == "__main__":
    unittest.main()
