# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=W0613, protected-access, too-many-arguments, too-many-locals
# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint raise error for the call method mocking
# pylint: disable=unnecessary-dunder-call

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

    def test_whenInstantiatingABPEmbSeq2SeqModel_thenShouldInstantiateAEmbeddingNetwork(
        self,
    ):
        seq2seq_model = BPEmbSeq2SeqModel(output_size=self.output_size)

        self.assertEqual(self.input_size, seq2seq_model.embedding_network.model.input_size)
        self.assertEqual(self.hidden_size, seq2seq_model.embedding_network.model.hidden_size)
        self.assertEqual(
            self.projection_size,
            seq2seq_model.embedding_network.projection_layer.out_features,
        )

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModel_thenEncoderIsCalledOnce(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        encoder_mock,
        torch_load_mock,
    ):
        seq2seq_model = BPEmbSeq2SeqModel(self.output_size)

        to_predict_mock, lengths_list = self.setup_encoder_mocks()
        encoder_mock.__call__().return_value = (MagicMock(), MagicMock())
        seq2seq_model._encoder_step(to_predict_mock, lengths_list, self.a_batch_size)

        encoder_call = [call()(to_predict_mock, lengths_list)]

        encoder_mock.assert_has_calls(encoder_call)

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModelNoTarget_thenDecoderIsCalled(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        decoder_mock,
        torch_load_mock,
    ):
        seq2seq_model = BPEmbSeq2SeqModel(output_size=self.output_size)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)

        view_mock = MagicMock()
        decoder_input_mock.view.return_value = view_mock

        encoder_outputs = MagicMock()
        seq2seq_model._decoder_step(
            decoder_input_mock,
            decoder_hidden_mock,
            encoder_outputs,
            self.a_none_target,
            self.a_lengths_list,
            self.a_batch_size,
        )

        decoder_call = [
            call()(view_mock, decoder_hidden_mock, encoder_outputs, self.a_lengths_list)
        ] * self.a_longest_sequence_length

        decoder_mock.assert_has_calls(decoder_call)

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqAttModelNoTarget_thenDecoderIsCalled(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        decoder_mock,
        torch_load_mock,
    ):
        seq2seq_model = BPEmbSeq2SeqModel(
            output_size=self.output_size,
            attention_mechanism=True,
        )

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=True)

        view_mock = MagicMock()
        decoder_input_mock.view.return_value = view_mock

        encoder_outputs = MagicMock()
        seq2seq_model._decoder_step(
            decoder_input_mock,
            decoder_hidden_mock,
            encoder_outputs,
            self.a_none_target,
            self.a_lengths_list,
            self.a_batch_size,
        )

        decoder_call = [
            call()(view_mock, decoder_hidden_mock, encoder_outputs, self.a_lengths_list)
        ] * self.a_longest_sequence_length

        decoder_mock.assert_has_calls(decoder_call)

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_whenInstantiateASeq2SeqModelWithTarget_thenDecoderIsCalled(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        decoder_mock,
        random_mock,
        torch_load_mock,
    ):
        random_mock.return_value = self.a_value_lower_than_threshold

        seq2seq_model = BPEmbSeq2SeqModel(output_size=self.output_size)

        decoder_input_mock, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)

        encoder_outputs = MagicMock()
        seq2seq_model._decoder_step(
            decoder_input_mock,
            decoder_hidden_mock,
            encoder_outputs,
            self.a_none_target,
            self.a_lengths_list,
            self.a_batch_size,
        )

        decoder_call = []

        for idx in range(self.a_longest_sequence_length):
            decoder_call.append(
                call()(
                    self.a_transpose_target_vector[idx].view(1, self.a_batch_size, 1),
                    decoder_hidden_mock,
                )
            )

        self.assert_has_calls_tensor_equals(decoder_mock, decoder_call)

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenABPEmbSeq2SeqModel_whenForwardPass_thenProperlyDoPAss(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        decoder_mock,
        encoder_mock,
        torch_load_mock,
    ):
        to_predict_mock, lengths_list = self.setup_encoder_mocks()

        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)
        decomposition_lengths_mock = MagicMock()

        # we don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        to_mock = MagicMock()
        torch_mock.zeros().new_full.return_value = to_mock

        # We mock the return of the decoder output
        encoder_mock.__call__().return_value = (decoder_input_mock, decoder_hidden_mock)

        with patch("deepparse.network.bpemb_seq2seq.EmbeddingNetwork") as embedding_network_patch:
            # we mock the output of the embedding layer
            embedded_output_mock = MagicMock()
            embedding_network_patch().return_value = embedded_output_mock
            with torch_mock:
                with encoder_mock:
                    with decoder_mock:
                        seq2seq_model = BPEmbSeq2SeqModel(self.fake_cache_dir, self.output_size)

                        seq2seq_model.forward(
                            to_predict=to_predict_mock,
                            decomposition_lengths=decomposition_lengths_mock,
                            lengths=lengths_list,
                            target=None,
                        )

                        embedding_network_patch.assert_has_calls([call()(to_predict_mock, decomposition_lengths_mock)])
                        encoder_mock.assert_has_calls([call()(embedded_output_mock, lengths_list)])

                        decoder_mock.assert_has_calls(
                            [
                                call()(
                                    to_mock,
                                    decoder_hidden_mock,
                                    decoder_input_mock,
                                    lengths_list,
                                )
                            ]
                        )

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenABPEmbSeq2SeqModel_whenForwardPassWithTarget_thenProperlyDoPAss(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        decoder_mock,
        encoder_mock,
        random_mock,
        torch_load_mock,
    ):
        random_mock.return_value = self.a_value_lower_than_threshold

        target_mock = MagicMock()
        to_predict_mock, lengths_list = self.setup_encoder_mocks()

        # 1) We reset it later
        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=False)
        decomposition_lengths_mock = MagicMock()

        # We don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        to_mock = MagicMock()
        torch_mock.zeros().new_full.return_value = to_mock

        # We mock the return of the decoder output
        encoder_mock.__call__().return_value = (decoder_input_mock, decoder_hidden_mock)

        with patch("deepparse.network.bpemb_seq2seq.EmbeddingNetwork") as embedding_network_patch:
            # we mock the output of the embedding layer
            embedded_output_mock = MagicMock()
            embedding_network_patch().return_value = embedded_output_mock
            seq2seq_model = BPEmbSeq2SeqModel(self.fake_cache_dir, self.output_size)

            seq2seq_model.forward(
                to_predict=to_predict_mock,
                decomposition_lengths=decomposition_lengths_mock,
                lengths=lengths_list,
                target=target_mock,
            )

            embedding_network_patch.assert_has_calls([call()(to_predict_mock, decomposition_lengths_mock)])
            encoder_mock.assert_has_calls([call()(embedded_output_mock, lengths_list)])

            decoder_mock.assert_has_calls(
                [
                    call()(
                        to_mock,
                        decoder_hidden_mock,
                        decoder_input_mock,
                        lengths_list,
                    )
                ]
            )
            target_mock.assert_has_calls([call.transpose(0, 1)])

    @patch("deepparse.weights_tools.torch")
    @patch("deepparse.network.seq2seq.random.random")
    @patch("deepparse.network.seq2seq.Encoder")
    @patch("deepparse.network.seq2seq.Decoder")
    @patch("os.path.isfile")
    @patch("deepparse.network.seq2seq.torch")
    @patch("deepparse.network.seq2seq.Seq2SeqModel.load_state_dict")
    def test_givenABPEmbAttSeq2SeqModel_whenForwardPassWithTarget_thenProperlyDoPAss(
        self,
        load_state_dict_mock,
        torch_mock,
        isfile_mock,
        decoder_mock,
        encoder_mock,
        random_mock,
        torch_load_mock,
    ):
        random_mock.return_value = self.a_value_lower_than_threshold

        target_mock = MagicMock()
        to_predict_mock, lengths_list = self.setup_encoder_mocks()

        _, decoder_hidden_mock = self.setUp_decoder_mocks(decoder_mock, attention_mechanism=True)
        decomposition_lengths_mock = MagicMock()

        # We don't use the one of the setUp_decoder_mocks since we do the full loop
        decoder_input_mock = MagicMock()
        to_mock = MagicMock()
        torch_mock.zeros().new_full.return_value = to_mock

        # We mock the return of the decoder output
        encoder_mock.__call__().return_value = (decoder_input_mock, decoder_hidden_mock)

        with patch("deepparse.network.bpemb_seq2seq.EmbeddingNetwork") as embedding_network_patch:
            # we mock the output of the embedding layer
            embedded_output_mock = MagicMock()
            embedding_network_patch().return_value = embedded_output_mock
            seq2seq_model = BPEmbSeq2SeqModel(
                self.fake_cache_dir,
                self.output_size,
                attention_mechanism=True,
            )

            seq2seq_model.forward(
                to_predict=to_predict_mock,
                decomposition_lengths=decomposition_lengths_mock,
                lengths=lengths_list,
                target=target_mock,
            )

            embedding_network_patch.assert_has_calls([call()(to_predict_mock, decomposition_lengths_mock)])
            encoder_mock.assert_has_calls([call()(embedded_output_mock, lengths_list)])

            decoder_mock.assert_has_calls(
                [
                    call()(
                        to_mock,
                        decoder_hidden_mock,
                        decoder_input_mock,
                        lengths_list,
                    )
                ]
            )
            target_mock.assert_has_calls([call.transpose(0, 1)])


if __name__ == "__main__":
    unittest.main()
