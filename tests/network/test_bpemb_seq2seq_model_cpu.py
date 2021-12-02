# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=W0613, protected-access, too-many-arguments, too-many-locals
# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import unittest
from unittest.mock import patch, call, MagicMock

import torch

from deepparse.network import BPEmbSeq2SeqModel
from tests.network.base import Seq2SeqTestCase


class BPEmbSeq2SeqCPUTest(Seq2SeqTestCase):

    @classmethod
    def setUpClass(cls):
        super(BPEmbSeq2SeqCPUTest, cls).setUpClass()
        cls.model_type = "bpemb"

        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=cls.a_cpu_device)
        cls.a_transpose_target_vector = cls.a_target_vector.transpose(0, 1)

    @patch("deepparse.network.seq2seq.Seq2SeqModel._load_pre_trained_weights")
    def test_whenInstantiatingABPEmbSeq2SeqModel_thenShouldInstantiateAEmbeddingNetwork(
            self, load_pre_trained_weights_mock):
        seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device, output_size=self.output_size, verbose=self.verbose)

        self.assertEqual(self.input_size, seq2seq_model.embedding_network.model.input_size)
        self.assertEqual(self.hidden_size, seq2seq_model.embedding_network.model.hidden_size)
        self.assertEqual(self.projection_size, seq2seq_model.embedding_network.projection_layer.out_features)

    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenNotLocalWeights_whenInstantiatingABPEmbSeq2SeqModel_thenShouldDownloadWeights(
            self, load_state_dict_mock, torch_mock, isfile_mock):
        isfile_mock.return_value = False
        with patch("deepparse.network.seq2seq.download_weights") as download_weights_mock:
            _ = BPEmbSeq2SeqModel(self.a_cpu_device, output_size=self.output_size, verbose=self.verbose)
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
            _ = BPEmbSeq2SeqModel(self.a_cpu_device, output_size=self.output_size, verbose=self.verbose)
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
            _ = BPEmbSeq2SeqModel(self.a_cpu_device, output_size=self.output_size, verbose=self.verbose)
            download_weights_mock.assert_not_called()

    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenRetrainedWeights_whenInstantiatingAFastTextSeq2SeqModel_thenShouldUseRetrainedWeights(
            self, load_state_dict_mock, torch_mock):
        all_layers_params = MagicMock()
        torch_mock.load.return_value = all_layers_params
        BPEmbSeq2SeqModel(self.a_cpu_device,
                          output_size=self.output_size,
                          verbose=self.verbose,
                          path_to_retrained_model=self.a_path_to_retrained_model)

        torch_load_call = [call.load(self.a_path_to_retrained_model, map_location=self.a_cpu_device)]
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
        seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device, self.output_size, self.verbose)

        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()
        encoder_mock.__call__().return_value = (MagicMock(), MagicMock())
        seq2seq_model._encoder_step(to_predict_mock, lengths_tensor_mock, self.a_batch_size)

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
        seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device, output_size=self.output_size, verbose=self.verbose)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)

        view_mock = MagicMock()
        decoder_input_mock.view.return_value = view_mock

        lengths_tensor_mock = MagicMock()
        max_length = 4  # a sequence of 4 tokens
        lengths_tensor_mock.max().item.return_value = max_length
        encoder_outputs = MagicMock()
        seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, encoder_outputs, self.a_none_target,
                                    lengths_tensor_mock, self.a_batch_size)

        decoder_call = [call()(view_mock, decoder_hidden_mock, encoder_outputs, lengths_tensor_mock)] * max_length

        decoder_mock.assert_has_calls(decoder_call)

    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqAttModelNoTarget_thenDecoderIsCalled(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        last_version_mock,
        download_weights_mock,
        decoder_mock,
    ):
        seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device,
                                          output_size=self.output_size,
                                          verbose=self.verbose,
                                          attention_mechanism=True)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=True)

        view_mock = MagicMock()
        decoder_input_mock.view.return_value = view_mock

        lengths_tensor_mock = MagicMock()
        max_length = 4  # a sequence of 4 tokens
        lengths_tensor_mock.max().item.return_value = max_length
        encoder_outputs = MagicMock()
        seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, encoder_outputs, self.a_none_target,
                                    lengths_tensor_mock, self.a_batch_size)

        decoder_call = [call()(view_mock, decoder_hidden_mock, encoder_outputs, lengths_tensor_mock)] * max_length

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

        seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device, output_size=self.output_size, verbose=self.verbose)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)

        lengths_tensor_mock = MagicMock()
        max_length = 4  # a sequence of 4 tokens
        lengths_tensor_mock.max().item.return_value = max_length
        encoder_outputs = MagicMock()
        seq2seq_model._decoder_step(decoder_input_mock, decoder_hidden_mock, encoder_outputs, self.a_none_target,
                                    lengths_tensor_mock, self.a_batch_size)

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
    def test_givenABPEmbSeq2SeqModel_whenForwardPass_thenProperlyDoPAss(self, load_state_dict_mock, torch_mock,
                                                                        isfile_mock, last_version_mock,
                                                                        download_weights_mock, decoder_mock,
                                                                        encoder_mock):
        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()

        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)
        decomposition_lengths_mock = MagicMock()

        # we don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        to_mock = MagicMock()
        torch_mock.zeros().to().new_full.return_value = to_mock

        # We mock the return of the decoder output
        encoder_mock.__call__().return_value = (decoder_input_mock, decoder_hidden_mock)

        with patch("deepparse.network.bpemb_seq2seq.EmbeddingNetwork") as embedding_network_patch:
            # we mock the output of the embedding layer
            embedded_output_mock = MagicMock()
            embedding_network_patch().return_value = embedded_output_mock
            with torch_mock:
                with encoder_mock:
                    with decoder_mock:
                        seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device, self.output_size, self.verbose)

                        seq2seq_model.forward(to_predict=to_predict_mock,
                                              decomposition_lengths=decomposition_lengths_mock,
                                              lengths_tensor=lengths_tensor_mock,
                                              target=None)

                        embedding_network_patch.assert_has_calls([call()(to_predict_mock, decomposition_lengths_mock)])
                        encoder_mock.assert_has_calls([call()(embedded_output_mock, lengths_tensor_mock)])
                        lengths_tensor_mock.assert_has_calls([call.max().item()])
                        decoder_mock.assert_has_calls(
                            [call()(to_mock, decoder_hidden_mock, decoder_input_mock, lengths_tensor_mock)])

    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenABPEmbSeq2SeqModel_whenForwardPassWithTarget_thenProperlyDoPAss(self, load_state_dict_mock,
                                                                                  torch_mock, isfile_mock,
                                                                                  last_version_mock,
                                                                                  download_weights_mock, decoder_mock,
                                                                                  encoder_mock, random_mock):
        random_mock.return_value = self.a_value_lower_than_threshold

        target_mock = MagicMock()
        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()

        # 1) We reset it later
        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)
        decomposition_lengths_mock = MagicMock()

        # We don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        to_mock = MagicMock()
        torch_mock.zeros().to().new_full.return_value = to_mock

        # We mock the return of the decoder output
        encoder_mock.__call__().return_value = (decoder_input_mock, decoder_hidden_mock)

        with patch("deepparse.network.bpemb_seq2seq.EmbeddingNetwork") as embedding_network_patch:
            # we mock the output of the embedding layer
            embedded_output_mock = MagicMock()
            embedding_network_patch().return_value = embedded_output_mock
            seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device, self.output_size, self.verbose)

            seq2seq_model.forward(to_predict=to_predict_mock,
                                  decomposition_lengths=decomposition_lengths_mock,
                                  lengths_tensor=lengths_tensor_mock,
                                  target=target_mock)

            embedding_network_patch.assert_has_calls([call()(to_predict_mock, decomposition_lengths_mock)])
            encoder_mock.assert_has_calls([call()(embedded_output_mock, lengths_tensor_mock)])
            lengths_tensor_mock.assert_has_calls([call.max().item()])
            decoder_mock.assert_has_calls(
                [call()(to_mock, decoder_hidden_mock, decoder_input_mock, lengths_tensor_mock)])
            target_mock.assert_has_calls([call.transpose(0, 1)])

    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("deepparse.network.seq2seq.download_weights")
    @patch("deepparse.network.seq2seq.latest_version")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenABPEmbAttSeq2SeqModel_whenForwardPassWithTarget_thenProperlyDoPAss(
            self, load_state_dict_mock, torch_mock, isfile_mock, last_version_mock, download_weights_mock, decoder_mock,
            encoder_mock, random_mock):
        random_mock.return_value = self.a_value_lower_than_threshold

        target_mock = MagicMock()
        to_predict_mock, lengths_tensor_mock = self.setup_encoder_mocks()

        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=True)
        decomposition_lengths_mock = MagicMock()

        # We don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        to_mock = MagicMock()
        torch_mock.zeros().to().new_full.return_value = to_mock

        # We mock the return of the decoder output
        encoder_mock.__call__().return_value = (decoder_input_mock, decoder_hidden_mock)

        with patch("deepparse.network.bpemb_seq2seq.EmbeddingNetwork") as embedding_network_patch:
            # we mock the output of the embedding layer
            embedded_output_mock = MagicMock()
            embedding_network_patch().return_value = embedded_output_mock
            seq2seq_model = BPEmbSeq2SeqModel(self.a_cpu_device,
                                              self.output_size,
                                              self.verbose,
                                              attention_mechanism=True)

            seq2seq_model.forward(to_predict=to_predict_mock,
                                  decomposition_lengths=decomposition_lengths_mock,
                                  lengths_tensor=lengths_tensor_mock,
                                  target=target_mock)

            embedding_network_patch.assert_has_calls([call()(to_predict_mock, decomposition_lengths_mock)])
            encoder_mock.assert_has_calls([call()(embedded_output_mock, lengths_tensor_mock)])
            lengths_tensor_mock.assert_has_calls([call.max().item()])
            decoder_mock.assert_has_calls(
                [call()(to_mock, decoder_hidden_mock, decoder_input_mock, lengths_tensor_mock)])
            target_mock.assert_has_calls([call.transpose(0, 1)])


if __name__ == "__main__":
    unittest.main()
