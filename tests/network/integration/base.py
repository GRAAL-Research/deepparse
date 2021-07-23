# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import os
import pickle
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from deepparse import download_from_url


class Seq2SeqIntegrationTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.verbose = False
        cls.a_torch_device = torch.device("cuda:0")
        cls.a_cpu_device = torch.device("cpu")
        cls.begin_of_sequence_idx = -1  # BOS
        cls.encoder_hidden_size = 1024

        cls.number_of_tags = 9  # default tag space of our models
        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=cls.a_torch_device)

        cls.output_size = 9

        cls.temp_dir_obj = TemporaryDirectory()
        cls.weights_dir = os.path.join(cls.temp_dir_obj.name, "./weights")

        download_from_url(file_name="to_predict_bpemb", saving_dir=cls.weights_dir, file_extension="p")
        download_from_url(file_name="to_predict_fasttext", saving_dir=cls.weights_dir, file_extension="p")
        download_from_url(file_name="decoder_hidden", saving_dir=cls.weights_dir, file_extension="p")

        cls.path = os.path.join(cls.temp_dir_obj.name, ".cache", "deepparse")
        cls.retrain_file_name_format = "retrained_{}_address_parser"

    @classmethod
    def models_setup(cls, model: str) -> None:
        # We download the "normal" model
        download_from_url(file_name=model, saving_dir=cls.path, file_extension="ckpt")

        # We download the "pre_trained" model
        model = cls.retrain_file_name_format.format(model)
        download_from_url(file_name=model, saving_dir=cls.path, file_extension="ckpt")
        cls.re_trained_output_dim = 3

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def encoder_input_setUp(self, model_type: str, device: torch.device):
        with open(os.path.join(self.weights_dir, f"to_predict_{model_type}.p"), "rb") as file:
            self.to_predict_tensor = pickle.load(file)
        self.to_predict_tensor = self.to_predict_tensor.to(device)

        self.a_lengths_tensor = torch.tensor([6, 6], device=device)
        self.a_batch_size = 2

    def encoder_output_setUp(self, device: torch.device):
        self.decoder_input = torch.tensor([[[-1.], [-1.]]], device=device)
        with open(os.path.join(self.weights_dir, "decoder_hidden.p"), "rb") as file:
            self.decoder_hidden_tensor = pickle.load(file)
        self.decoder_hidden_tensor = (self.decoder_hidden_tensor[0].to(device),
                                      self.decoder_hidden_tensor[1].to(device))

    def decoder_input_setUp(self):
        self.max_length = self.a_lengths_tensor[0].item()

    def assert_output_is_valid_dim(self, actual_prediction, output_dim):
        self.assertEqual(self.max_length + 1, actual_prediction.shape[0])  # + 1 since end-of-sequence (EOS)
        self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
        self.assertEqual(output_dim, actual_prediction.shape[2])
