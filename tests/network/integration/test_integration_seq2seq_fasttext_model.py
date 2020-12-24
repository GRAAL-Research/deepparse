# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access
import unittest
from unittest import skipIf

import os
import torch

from deepparse.network import FastTextSeq2SeqModel
from ..integration.base import Seq2SeqIntegrationTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class FastTextSeq2SeqIntegrationTest(Seq2SeqIntegrationTestCase):

    def setUp(self) -> None:
        super().setUp()
        # will load the weights if not local
        self.encoder_input_setUp("fasttext")

        self.a_retrain_model = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.ckpt")

    def test_whenForwardStep_thenStepIsOk(self):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device)
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)

    def test_whenForwardStepWithTarget_thenStepIsOk(self):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device)
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_tensor, self.a_target_vector)

        self.assert_output_is_valid_dim(predictions)

    def test_retrainedModel_whenForwardStep_thenStepIsOk(self):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device,
                                                  self.verbose,
                                                  path_to_retrained_model=self.a_retrain_model)
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_tensor)

        self.assert_output_is_valid_dim(predictions)

    def test_retrainedModel_whenForwardStepWithTarget_thenStepIsOk(self):
        self.seq2seq_model = FastTextSeq2SeqModel(self.a_torch_device,
                                                  self.verbose,
                                                  path_to_retrained_model=self.a_retrain_model)
        # forward pass for two address: '['15 major st london ontario n5z1e1', '15 major st london ontario n5z1e1']'
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_tensor, self.a_target_vector)

        self.assert_output_is_valid_dim(predictions)


if __name__ == "__main__":
    unittest.main()
