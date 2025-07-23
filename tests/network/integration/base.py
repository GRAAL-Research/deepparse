# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
import safetensors
from transformers.utils.hub import cached_file

from deepparse import download_weights


class Seq2SeqIntegrationTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verbose = False
        cls.begin_of_sequence_idx = -1  # BOS
        cls.encoder_hidden_size = 1024
        cls.decoder_hidden_size = 1024
        cls.input_size = 300
        cls.num_layers = 1

        cls.number_of_tags = 9  # default tag space of our models

        cls.a_batch_size = 2
        cls.sequence_len = 6
        cls.decomposition_len = 6

        cls.output_size = 9

        cls.temp_dir_obj = TemporaryDirectory()
        cls.weights_dir = os.path.join(cls.temp_dir_obj.name, "./weights")

        cls.path = os.path.join(cls.temp_dir_obj.name, ".cache", "deepparse")
        cls.retrain_file_name_format = "retrained_{}_address_parser.ckpt"

        cls.cache_dir = cls.path  # We use the same cache dir as the path we download the models and weights

    @classmethod
    def models_setup(cls, model_type: str, cache_dir: str) -> None:
        # We download the "normal" model and the .version file
        model_id = download_weights(model_type, cache_dir, verbose=False, offline=False)

        # We also simulate a retrained model
        model_file_name = cls.retrain_file_name_format.format(model_type)

        weights = safetensors.torch.load_file(
            cached_file(model_id, filename="model.safetensors", local_files_only=True, cache_dir=cache_dir)
        )

        version = "Finetuned_"
        torch_save = {
            "address_tagger_model": weights,
            "model_type": model_type,
            "version": version,
            "named_parser": "SimulatedRetrainedParser",
        }
        torch.save(torch_save, os.path.join(cache_dir, model_file_name))

        cls.re_trained_output_dim = 3

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def encoder_input_setUp(self, model_type: str, device: torch.device):
        if "bpemb" in model_type:
            self.to_predict_tensor = torch.rand(
                (self.a_batch_size, self.sequence_len, self.decomposition_len, self.input_size)
            )
        else:
            self.to_predict_tensor = torch.rand((self.a_batch_size, self.sequence_len, self.input_size))

        self.to_predict_tensor = self.to_predict_tensor.to(device)

        self.a_lengths_list = [6, 6]
        self.a_batch_size = 2

    def encoder_output_setUp(self, device: torch.device):
        self.decoder_input = torch.tensor([[[-1.0], [-1.0]]], device=device)

        self.decoder_hidden_tensor = (
            torch.rand((self.num_layers, self.a_batch_size, self.decoder_hidden_size)).to(device),
            torch.rand((self.num_layers, self.a_batch_size, self.decoder_hidden_size)).to(device),
        )
        self.encoder_hidden = torch.rand((self.a_batch_size, self.a_target_vector.shape[1], self.encoder_hidden_size))

    def decoder_input_setUp(self):
        self.a_longest_sequence_length = self.a_lengths_list[0]

    def assert_output_is_valid_dim(self, actual_prediction, output_dim):
        self.assertEqual(
            self.a_longest_sequence_length + 1, actual_prediction.shape[0]
        )  # + 1 since end-of-sequence (EOS)
        self.assertEqual(self.a_batch_size, actual_prediction.shape[1])
        self.assertEqual(output_dim, actual_prediction.shape[2])
