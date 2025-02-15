# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

# Pylint raise error for the call method mocking
# pylint: disable=unnecessary-dunder-call

# Pylint kick for temporary directory
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from tests.tools import create_file


class Seq2SeqTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_cpu_device = torch.device("cpu")
        cls.a_torch_device = torch.device("cuda:0")
        cls.verbose = False
        cls.a_batch_size = 2
        cls.a_none_target = None
        cls.a_value_lower_than_threshold = 0.1

        cls.a_path_to_retrained_model = "a/path/to/a/retrain/model"

        cls.input_size = 300
        cls.hidden_size = 300
        cls.projection_size = 300

        cls.output_size = 9

        cls.temp_dir_obj = TemporaryDirectory()
        cls.fake_cache_dir = os.path.join(cls.temp_dir_obj.name, "fake_cache")
        os.makedirs(cls.fake_cache_dir, exist_ok=True)

        cls.a_fasttext_hash_value = "f67a0517c70a314bdde0b8440f21139d"
        create_file(os.path.join(cls.fake_cache_dir, "fasttext.version"), cls.a_fasttext_hash_value)
        create_file(os.path.join(cls.fake_cache_dir, "fasttext_attention.version"), cls.a_fasttext_hash_value)

        cls.a_bpemb_hash_value = "f67a0517c70a314bdde0b8440f21139d"
        create_file(os.path.join(cls.fake_cache_dir, "bpemb.version"), cls.a_bpemb_hash_value)
        create_file(os.path.join(cls.fake_cache_dir, "bpemb_attention.version"), cls.a_bpemb_hash_value)

        # A batch of 3 sequence of length (respectively): 2, 4 and 3. Thus, the longest one is 4.
        cls.a_lengths_list = [1, 4, 3]
        cls.a_longest_sequence_length = 4  # base on the "a_lengths_list" the longest is 4.

    def setup_encoder_mocks(self):
        to_predict_mock = MagicMock()

        return to_predict_mock, self.a_lengths_list

    def setUp_decoder_mocks(self, decoder_mock, attention_mechanism):
        decoder_input_mock = MagicMock()
        decoder_hidden_mock = MagicMock()

        attention_weights = MagicMock() if attention_mechanism else None

        decoder_output = MagicMock()
        decoder_output.topk.return_value = MagicMock(), decoder_input_mock
        decoder_mock.__call__().return_value = (
            decoder_output,
            decoder_hidden_mock,
            attention_weights,
        )

        return decoder_input_mock, decoder_hidden_mock

    def assert_has_calls_tensor_equals(self, decoder_mock, expected_calls):
        # since we can"t compare tensor in calls, we open it and compare each elements
        decoder_mock_calls = decoder_mock.mock_calls[8:11]
        for decoder_mock_call, expected_call in zip(decoder_mock_calls, expected_calls):
            self.assertEqual(decoder_mock_call[1][0].tolist(), expected_call[1][0].tolist())  # the tensor
            self.assertEqual(decoder_mock_call[1][1], expected_call[1][1])  # The other element of the call
