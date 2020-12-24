# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=W0613, protected-access, too-many-arguments
import unittest
from unittest.mock import patch, call, MagicMock

from deepparse.network import BPEmbSeq2SeqModel
from tests.network.base import Seq2SeqTestCase


class BPEmbSeq2SeqTest(Seq2SeqTestCase):

    def setUp(self) -> None:
        self.input_size = 300
        self.hidden_size = 300
        self.projection_size = 300
        self.model_type = "bpemb"

    @patch("deepparse.network.seq2seq.Seq2SeqModel._load_pre_trained_weights")
    def test_whenInstantiatingABPEmbSeq2SeqModel_thenShouldInstantiateAEmbeddingNetwork(
            self, load_pre_trained_weights_mock):
        self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)

        self.assertEqual(self.input_size, self.seq2seq_model.embedding_network.model.input_size)
        self.assertEqual(self.hidden_size, self.seq2seq_model.embedding_network.model.hidden_size)
        self.assertEqual(self.projection_size, self.seq2seq_model.embedding_network.projection_layer.out_features)

    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenNotLocalWeights_whenInstantiatingABPEmbSeq2SeqModel_thenShouldDownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock):
        isfile_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenLocalWeightsNotLastVersion_whenInstantiatingABPEmbSeq2SeqModel_thenShouldDownloadWeights(
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
    def test_givenLocalWeights_whenInstantiatingABPEmbSeq2SeqModel_thenShouldntDownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = True
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)
            download_weights_mock.assert_not_called()

    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenRetrainedWeights_whenInstantiatingAFastTextSeq2SeqModel_thenShouldUseRetrainedWeights(
            self, load_state_dict_mock, torch_mock):
        all_layers_params = MagicMock()
        torch_mock.load.return_value = all_layers_params
        self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device,
                                               self.verbose,
                                               path_to_retrained_model=self.a_path_to_retrained_model)

        torch_load_call = [call.load(self.a_path_to_retrained_model, map_location=self.a_torch_device)]
        torch_mock.assert_has_calls(torch_load_call)

        load_state_dict_call = [call(all_layers_params)]
        load_state_dict_mock.assert_has_calls(load_state_dict_call)

    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModel_thenEncoderIsCalledOnce(self, load_state_dict_mock, torch_mock, isfile_mock,
                                                                  last_version_mock, download_weights_mock,
                                                                  encoder_mock):
        self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)

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
    def test_whenInstantiateASeq2SeqModelNoTarget_thenDecoderIsCalled(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        last_version_mock,
        download_weights_mock,
        decoder_mock,
    ):
        self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)

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
                                                                        download_weights_mock, decoder_mock,
                                                                        random_mock):
        random_mock.return_value = self.a_value_lower_than_threshold

        self.seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device, self.verbose)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock)
        max_length = 4  # a sequence of 4 tokens
        self.seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, self.a_target_vector, max_length,
                                         self.a_batch_size)

        decoder_call = []

        for idx in range(max_length):
            decoder_call.append(call()(self.a_transpose_target_vector[idx].view(1, self.a_batch_size, 1),
                                       decoder_hidden_mock))

        self.assert_has_calls_tensor_equals(decoder_mock, decoder_call)


if __name__ == "__main__":
    unittest.main()
