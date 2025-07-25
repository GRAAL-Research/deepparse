# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, protected-access

import os
import unittest
from unittest import skipIf

import torch

from deepparse.network import BPEmbSeq2SeqModel
from ..integration.base import Seq2SeqIntegrationTestCase


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class BPEmbSeq2SeqIntegrationTest(Seq2SeqIntegrationTestCase):
    @classmethod
    def setUpClass(cls):
        super(BPEmbSeq2SeqIntegrationTest, cls).setUpClass()
        cls.a_torch_device = torch.device("cuda:0")
        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=cls.a_torch_device)
        cls.models_setup(model_type="bpemb", cache_dir=cls.path)
        cls.a_retrain_model_path = os.path.join(cls.path, cls.retrain_file_name_format.format("bpemb") + ".ckpt")

    def setUp(self) -> None:
        # will load the weights if not local
        self.encoder_input_setUp("bpemb", self.a_torch_device)
        self.decomposition_lengths = [[1, 1, 1, 1, 1, 6], [1, 1, 1, 1, 1, 6]]

        self.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=self.a_torch_device)

    def test_whenForwardStep_thenStepIsOk(self):
        self.seq2seq_model = BPEmbSeq2SeqModel(output_size=self.number_of_tags)
        self.seq2seq_model.to_device(self.a_torch_device)

        # forward pass for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(
            self.to_predict_tensor, self.decomposition_lengths, self.a_lengths_list
        )

        self.assert_output_is_valid_dim(predictions, output_dim=self.number_of_tags)

    def test_whenForwardStepWithTarget_thenStepIsOk(self):
        self.seq2seq_model = BPEmbSeq2SeqModel(output_size=self.number_of_tags)
        self.seq2seq_model.to_device(self.a_torch_device)

        # forward pass for two address: "["15 major st london ontario n5z1e1", "15 major st london ontario n5z1e1"]"
        self.decoder_input_setUp()

        predictions = self.seq2seq_model.forward(
            self.to_predict_tensor,
            self.decomposition_lengths,
            self.a_lengths_list,
            self.a_target_vector,
        )

        self.assert_output_is_valid_dim(predictions, output_dim=self.number_of_tags)


if __name__ == "__main__":
    unittest.main()
