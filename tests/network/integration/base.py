# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

import pickle
from unittest import TestCase

import torch


class Seq2SeqIntegrationTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.verbose = False
        cls.a_torch_device = torch.device("cuda:0")
        cls.begin_of_sequence_idx = -1  # BOS
        cls.encoder_hidden_size = 1024
        cls.number_of_tags = 9  # tag space of our models
        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=cls.a_torch_device)

    def encoder_input_setUp(self, model_type: str):
        file = open(f"./tests/network/integration/to_predict_{model_type}.p", "rb")
        self.to_predict_tensor = pickle.load(file)
        self.to_predict_tensor = self.to_predict_tensor.to(self.a_torch_device)
        file.close()

        self.a_lengths_tensor = torch.tensor([6, 6], device=self.a_torch_device)
        self.a_batch_size = 2

    def encoder_output_setUp(self):
        self.decoder_input = torch.tensor([[[-1.], [-1.]]], device=self.a_torch_device)
        file = open("./tests/network/integration/decoder_hidden.p", "rb")
        self.decoder_hidden_tensor = pickle.load(file)
        self.decoder_hidden_tensor = (self.decoder_hidden_tensor[0].to(self.a_torch_device),
                                      self.decoder_hidden_tensor[1].to(self.a_torch_device))
        file.close()

    def decoder_input_setUp(self):
        self.max_length = self.a_lengths_tensor[0].item()

    def assert_output_is_valid_dim(self, actual_prediction):
        self.assertEqual(self.max_length + 1, actual_prediction.shape[0])  # + 1 since end-of-sequence (EOS)
        self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
        self.assertEqual(self.number_of_tags, actual_prediction.shape[2])
