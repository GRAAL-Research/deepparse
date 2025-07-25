# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access

import os
import unittest
from unittest import skipIf

import torch

from deepparse.network import FastTextSeq2SeqModel
from ..integration.base import Seq2SeqIntegrationTestCase


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class FastTextSeq2SeqIntegrationTest(Seq2SeqIntegrationTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastTextSeq2SeqIntegrationTest, cls).setUpClass()
        cls.a_cpu_device = torch.device("cpu")
        cls.models_setup(model_type="fasttext", cache_dir=cls.path)
        cls.a_retrain_model_path = os.path.join(cls.path, cls.retrain_file_name_format.format("fasttext") + ".ckpt")

    def setUp(self) -> None:
        super().setUp()
        # will load the weights if not local
        self.encoder_input_setUp("fasttext", self.a_cpu_device)

        self.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=self.a_cpu_device)

    def test_whenForwardStep_thenStepIsOk(self):
        self.seq2seq_model = FastTextSeq2SeqModel(output_size=self.number_of_tags)
        # forward pass for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_list)

        self.assert_output_is_valid_dim(predictions, output_dim=self.number_of_tags)

    def test_whenForwardStepWithTarget_thenStepIsOk(self):
        self.seq2seq_model = FastTextSeq2SeqModel(output_size=self.number_of_tags)
        # forward pass for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(self.to_predict_tensor, self.a_lengths_list, self.a_target_vector)

        self.assert_output_is_valid_dim(predictions, output_dim=self.number_of_tags)


if __name__ == "__main__":
    unittest.main()
