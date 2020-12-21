# Since we use patch we skip the unused argument error
# pylint: disable=W0613

from unittest import TestCase
from unittest.mock import patch

import os
import torch

from deepparse.network import BPEmbSeq2SeqModel


class BPEmbSeq2SeqTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_type = "bpemb"
        cls.a_torch_device = torch.device("cpu")
        cls.verbose = False

        cls.input_size = 300
        cls.hidden_size = 300
        cls.projection_size = 300

    def setUp(self) -> None:
        self.a_root_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

    @patch("deepparse.network.seq2seq.Seq2SeqModel._load_pre_trained_weights")
    def test_InstantiateABPEmbSeq2SeqModel_InstantiateAEmbeddingNetwork(self, load_pre_trained_weights_mock):
        self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)

        self.assertEqual(self.input_size, self.seq2seq_model.embedding_network.model.input_size)
        self.assertEqual(self.hidden_size, self.seq2seq_model.embedding_network.model.hidden_size)
        self.assertEqual(self.projection_size, self.seq2seq_model.embedding_network.projection_layer.out_features)

    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_NotLocalWeights_InstantiateABPEmbSeq2SeqModel_DownloadWeights(self, load_state_dict_mock, torch_mock,
                                                                           isfile_mock):
        isfile_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_LocalWeightsNotLastVersion_InstantiateABPEmbSeq2SeqModel_DownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_LocalWeights_InstantiateABPEmbSeq2SeqModel_DontDownloadWeights(self, load_state_dict_mock, torch_mock,
                                                                            isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = True
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_not_called()
