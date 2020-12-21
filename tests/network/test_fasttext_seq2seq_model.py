# Since we use patch we skip the unused argument error
# pylint: disable=W0613

from unittest import TestCase
from unittest.mock import patch, MagicMock, call

import os
import torch

from deepparse.network import FastTextSeq2SeqModel


class FasttextSeq2SeqTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_type = "fasttext"
        cls.a_torch_device = torch.device("cpu")
        cls.verbose = False
        cls.a_batch_size = 2
        cls.a_none_target = None
        cls.a_value_lower_than_threshold = 0.1
        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]],
                                           device=cls.a_torch_device)
        cls.a_transpose_target_vector = cls.a_target_vector.transpose(0, 1)

    def setUp(self) -> None:
        self.a_root_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

    def setup_encoder_mocks(self):
        to_predict_mock = MagicMock()
        lengths_tensor_mock = MagicMock()
        return to_predict_mock, lengths_tensor_mock

    def setUp_decoder_mocks(self, decoder_mock):
        decoder_input_mock = MagicMock()
        decoder_hidden_mock = MagicMock()

        decoder_output = MagicMock()
        decoder_output.topk.return_value = MagicMock(), decoder_input_mock
        decoder_mock().return_value = decoder_output, decoder_hidden_mock

        return decoder_input_mock, decoder_hidden_mock

    def assert_has_calls_tensor_equals(self, decoder_mock, expected_calls):
        # since we can't compare tensor in calls, we open it and compare each elements
        decoder_mock_calls = decoder_mock.mock_calls[4:8]
        for decoder_mock_call, expected_call in zip(decoder_mock_calls, expected_calls):
            self.assertEqual(decoder_mock_call[1][0].tolist(), expected_call[1][0].tolist())  # the tensor
            self.assertEqual(decoder_mock_call[1][1], expected_call[1][1])  # The other element of the call

    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_NotLocalWeights_InstantiateAFastTextSeq2SeqModel_DownloadWeights(self, load_state_dict_mock, torch_mock,
                                                                              isfile_mock):
        isfile_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_LocalWeightsNotLastVersion_InstantiateAFastTextSeq2SeqModel_DownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_LocalWeights_InstantiateAFastTextSeq2SeqModel_DontDownloadWeights(self, load_state_dict_mock, torch_mock,
                                                                               isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = True
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_not_called()

    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModel_thenEncodeIsCalledOnce(self, load_state_dict_mock, torch_mock,
                                                                 isfile_mock, last_version_mock, download_weights_mock,
                                                                 encoder_mock):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.verbose)

        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()
        self.seq2seq_model._encoder_step(to_predict_mock, lengths_tensor_mock, self.a_batch_size)

        encoder_call = [call()(to_predict_mock, lengths_tensor_mock)]

        encoder_mock.assert_has_calls(encoder_call)

    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModelNoTarget_thenDecoderIsCalled(self, load_state_dict_mock, torch_mock,
                                                                      isfile_mock, last_version_mock,
                                                                      download_weights_mock,
                                                                      decoder_mock, ):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.verbose)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock)
        max_length = 4  # a sequence of 4 tokens
        self.seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, self.a_none_target, max_length,
                                         self.a_batch_size)

        decoder_call = [call()(decoder_input_mock.view(), decoder_hidden_mock)] * max_length

        decoder_mock.assert_has_calls(decoder_call)

    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModelWithTarget_thenDecoderIsCalled(self, load_state_dict_mock, torch_mock,
                                                                        isfile_mock, last_version_mock,
                                                                        download_weights_mock,
                                                                        decoder_mock, random_mock):
        random_mock.return_value = self.a_value_lower_than_threshold

        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.verbose)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock)
        max_length = 4  # a sequence of 4 tokens
        self.seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, self.a_target_vector, max_length,
                                         self.a_batch_size)

        decoder_call = []

        for idx in range(max_length):
            decoder_call.append(
                call()(self.a_transpose_target_vector[idx].view(1, self.a_batch_size, 1), decoder_hidden_mock))

        self.assert_has_calls_tensor_equals(decoder_mock, decoder_call)
