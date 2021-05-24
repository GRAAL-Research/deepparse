# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the encoder and decoder step
# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=unused-argument, protected-access, too-many-arguments, not-callable, too-many-locals

import unittest
from unittest import skipIf
from unittest.mock import patch, call, MagicMock

import torch

from deepparse.network import FastTextSeq2SeqModel
from tests.network.base import Seq2SeqTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class FasttextSeq2SeqGPUTest(Seq2SeqTestCase):

    @classmethod
    def setUpClass(cls):
        super(FasttextSeq2SeqGPUTest, cls).setUpClass()
        cls.model_type = "fasttext"

        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=cls.a_torch_device)
        cls.a_transpose_target_vector = cls.a_target_vector.transpose(0, 1)

    @patch("deepparse.network.seq2seq.Seq2SeqModel._load_pre_trained_weights")
    def test_whenInstantiatingAFasttextSeq2SeqModel_thenShouldInstantiateAEmbeddingNetwork(
            self, load_pre_trained_weights_mock):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)

        load_pre_trained_weights_mock.assert_called_with(self.model_type)

    @patch("deepparse.network.seq2seq.Seq2SeqModel._load_weights")
    def test_whenInstantiatingAFasttextSeq2SeqModelWithPath_thenShouldCallLoadWeights(self, load_weights_mock):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose,
                                                  self.a_path_to_retrained_model)

        load_weights_mock.assert_called_with(self.a_path_to_retrained_model)

    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenNotLocalWeights_whenInstantiatingAFastTextSeq2SeqModel_thenShouldDownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock):
        isfile_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenLocalWeightsNotLastVersion_whenInstantiatingAFastTextSeq2SeqModel_thenShouldDownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock):
        isfile_mock.return_value = True
        last_version_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)
            download_weights_mock.assert_called_with(self.model_type, self.a_root_path, verbose=self.verbose)

    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenRetrainedWeights_whenInstantiatingAFastTextSeq2SeqModel_thenShouldUseRetrainedWeights(
            self, load_state_dict_mock, torch_mock):
        all_layers_params = MagicMock()
        torch_mock.load.return_value = all_layers_params
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device,
                                                  self.output_size,
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
    def test_whenInstantiateASeq2SeqModel_thenEncodeIsCalledOnce(self, load_state_dict_mock, torch_mock, isfile_mock,
                                                                 last_version_mock, download_weights_mock,
                                                                 encoder_mock):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)

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
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)

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

        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock)
        max_length = 4  # a sequence of 4 tokens
        self.seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, self.a_target_vector, max_length,
                                         self.a_batch_size)

        decoder_call = []

        for idx in range(max_length):
            decoder_call.append(call()(self.a_transpose_target_vector[idx].view(1, self.a_batch_size, 1),
                                       decoder_hidden_mock))

        self.assert_has_calls_tensor_equals(decoder_mock, decoder_call)

    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenAFasttextSeq2SeqModel_whenForwardPass_thenProperlyDoPAss(self, load_state_dict_mock, torch_mock,
                                                                           isfile_mock, last_version_mock,
                                                                           download_weights_mock, decoder_mock,
                                                                           encoder_mock):
        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()

        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock)

        # we don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        torch_mock.zeros().to().new_full.return_value = decoder_input_mock

        with torch_mock:
            with encoder_mock:
                # we mock the return of the decoder output
                encoder_mock().return_value = decoder_hidden_mock
                with decoder_mock:
                    seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)

                    seq2seq_model.forward(to_predict=to_predict_mock, lengths_tensor=lengths_tensor_mock, target=None)

                    encoder_mock.assert_has_calls([call()(to_predict_mock, lengths_tensor_mock)])
                    lengths_tensor_mock.assert_has_calls([call.max().item()])
                    decoder_mock.assert_has_calls([call()(decoder_input_mock, decoder_hidden_mock)])

    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenAFasttext2SeqModel_whenForwardPassWithTarget_thenProperlyDoPAss(self, load_state_dict_mock,
                                                                                  torch_mock, isfile_mock,
                                                                                  last_version_mock,
                                                                                  download_weights_mock, decoder_mock,
                                                                                  encoder_mock, random_mock):
        random_mock.return_value = self.a_value_lower_than_threshold

        target_mock = MagicMock()
        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()

        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock)

        # we don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        torch_mock.zeros().to().new_full.return_value = decoder_input_mock

        with torch_mock:
            with encoder_mock:
                # we mock the return of the decoder output
                encoder_mock().return_value = decoder_hidden_mock
                with decoder_mock:
                    seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device, self.output_size, self.verbose)

                    seq2seq_model.forward(to_predict=to_predict_mock,
                                          lengths_tensor=lengths_tensor_mock,
                                          target=target_mock)

                    encoder_mock.assert_has_calls([call()(to_predict_mock, lengths_tensor_mock)])
                    lengths_tensor_mock.assert_has_calls([call.max().item()])
                    decoder_mock.assert_has_calls([call()(decoder_input_mock, decoder_hidden_mock)])
                    target_mock.assert_has_calls([call.transpose(0, 1)])


if __name__ == "__main__":
    unittest.main()
