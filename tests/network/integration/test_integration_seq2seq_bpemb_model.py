# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access

from unittest import skipIf

import torch

from deepparse.network import BPEmbSeq2SeqModel
from ..integration.base import Seq2SeqIntegrationTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class BPEmbSeq2SeqIntegrationTest(Seq2SeqIntegrationTestCase):

    def setUp(self) -> None:
        # will load the weights if not local
        self.pre_trained_seq2seq_model = BPEmbSeq2SeqModel(self.a_torch_device)
        self.encoder_input_setUp("bpemb")
        self.decomposition_lengths = [[1, 1, 1, 1, 1, 6], [1, 1, 1, 1, 1, 6]]

    def test_whenForwardStep_thenStepIsOk(self):
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.pre_trained_seq2seq_model.forward(self.to_predict_tensor, self.decomposition_lengths,
                                                             self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)

    def test_whenForwardStepWithTarget_thenStepIsOk(self):
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.pre_trained_seq2seq_model.forward(self.to_predict_tensor, self.decomposition_lengths,
                                                             self.a_lengths_tensor, self.a_target_vector)

        self.assert_output_is_valid_dim(predictions)

    # todo test for loading pretrained
