# Since we use patch we skip the unused argument error
# pylint: disable=W0613

from unittest import TestCase
from unittest.mock import patch

import os
import torch

from deepparse.network import PreTrainedFastTextSeq2SeqModel


class PreTrainedFasttextSeq2SeqTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_type = "fasttext"
        cls.a_torch_device = torch.device("cpu")
        cls.verbose = False

    def setUp(self) -> None:
        self.a_root_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

    @patch("os.path.isfile")
    @patch("deepparse.network.pre_trained_seq2seq.torch")
    @patch("deepparse.network.pre_trained_seq2seq.PreTrainedSeq2SeqModel.load_state_dict")
    def test_NotLocalWeights_InstantiateAFastTextPreTrainedSeq2SeqModel_DownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock):
        isfile_mock.return_value = False
        with patch("deepparse.network.pre_trained_seq2seq.download_weights") as download_weights_mock:
            self.pre_trained_seq2seq_model = PreTrainedFastTextSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.pre_trained_seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.pre_trained_seq2seq.torch")
    @patch("deepparse.network.pre_trained_seq2seq.PreTrainedSeq2SeqModel.load_state_dict")
    def test_LocalWeightsNotLastVersion_InstantiateAFastTextPreTrainedSeq2SeqModel_DownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.pre_trained_seq2seq.download_weights") as download_weights_mock:
            self.pre_trained_seq2seq_model = PreTrainedFastTextSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.pre_trained_seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.pre_trained_seq2seq.torch")
    @patch("deepparse.network.pre_trained_seq2seq.PreTrainedSeq2SeqModel.load_state_dict")
    def test_LocalWeights_InstantiateAFastTextPreTrainedSeq2SeqModel_DontDownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = True
        with patch("deepparse.network.pre_trained_seq2seq.download_weights") as download_weights_mock:
            self.pre_trained_seq2seq_model = PreTrainedFastTextSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_not_called()
