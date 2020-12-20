# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access

from unittest import skipIf

import torch

from deepparse.network import PreTrainedSeq2SeqModel, PreTrainedFastTextSeq2SeqModel, PreTrainedBPEmbSeq2SeqModel
from tests.network.base import PreTrainedSeq2SeqTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class PreTrainedSeq2SeqTest(PreTrainedSeq2SeqTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.pre_trained_seq2seq_model = PreTrainedSeq2SeqModel(self.a_torch_device)
        self.encoder_input_setUp("fasttext")  # fasttext since the simplest case (bpemb use a embedding layer)

    def test_whenEncoderStep_thenEncoderStepIsOk(self):
        # encoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'

        actual_decoder_input, actual_decoder_hidden = self.pre_trained_seq2seq_model._encoder_step(
            self.to_predict_tensor, self.a_lengths_tensor, self.a_batch_size)

        self.assertEqual(self.a_batch_size, actual_decoder_input.shape[1])
        self.assertTrue(actual_decoder_input[0][0] == self.begin_of_sequence_idx)
        self.assertTrue(actual_decoder_input[0][1] == self.begin_of_sequence_idx)

        self.assertEqual(self.a_batch_size, len(actual_decoder_hidden))
        self.assertEqual(self.encoder_hidden_size, actual_decoder_hidden[0].shape[2])
        self.assertEqual(self.encoder_hidden_size, actual_decoder_hidden[1].shape[2])

    def test_whenDecoderStep_thenDecoderStepIsOk(self):
        # decoding for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.encoder_output_setUp()
        self.decoder_input_setUp()

        actual_prediction_sequence = self.pre_trained_seq2seq_model._decoder_steps(self.decoder_input,
                                                                                   self.decoder_hidden_tensor,
                                                                                   self.max_length, self.a_batch_size)

        self.assert_output_is_valid_dim(actual_prediction_sequence)


@skipIf(not torch.cuda.is_available(), "no gpu available")
class PreTrainedFastTextSeq2SeqTest(PreTrainedSeq2SeqTestCase):

    def setUp(self) -> None:
        # will load the weights if not local
        self.pre_trained_seq2seq_model = PreTrainedFastTextSeq2SeqModel(self.a_torch_device)
        self.encoder_input_setUp("fasttext")

    def test_whenForwardStep_thenStepIsOk(self):
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.pre_trained_seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)


@skipIf(not torch.cuda.is_available(), "no gpu available")
class PreTrainedBPEmbSeq2SeqTest(PreTrainedSeq2SeqTestCase):

    def setUp(self) -> None:
        # will load the weights if not local
        self.pre_trained_seq2seq_model = PreTrainedBPEmbSeq2SeqModel(self.a_torch_device)
        self.encoder_input_setUp("bpemb")
        self.decomposition_lengths = [[1, 1, 1, 1, 1, 6], [1, 1, 1, 1, 1, 6]]

    def test_whenForwardStep_thenStepIsOk(self):
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.pre_trained_seq2seq_model.forward(
            self.to_predict_tensor,
            self.decomposition_lengths,
            self.a_lengths_tensor,
        )

        self.assert_output_is_valid_dim(predictions)
