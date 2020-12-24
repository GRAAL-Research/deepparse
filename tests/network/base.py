# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

import os
from unittest import TestCase
from unittest.mock import MagicMock

import torch


class Seq2SeqTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_torch_device = torch.device("cpu")
        cls.verbose = False
        cls.a_batch_size = 2
        cls.a_none_target = None
        cls.a_value_lower_than_threshold = 0.1
        cls.a_target_vector = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=cls.a_torch_device)
        cls.a_transpose_target_vector = cls.a_target_vector.transpose(0, 1)

        cls.a_root_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

        cls.a_path_to_retrained_model = "a/path/to/a/retrain/model"

    def setup_encoder_mocks(self):
        to_predict_mock = MagicMock()
        lengths_tensor_mock = MagicMock()
        return to_predict_mock, lengths_tensor_mock

    def setUp_decoder_mocks(self, decoder_mock):
        decoder_input_mock = MagicMock()
        decoder_hidden_mock = MagicMock()

        decoder_output = MagicMock()
        decoder_output.topk.return_value = MagicMock(), decoder_input_mock
        decoder_mock().return_value = decoder_output, decoder_hidden_mock

        return decoder_input_mock, decoder_hidden_mock

    def assert_has_calls_tensor_equals(self, decoder_mock, expected_calls):
        # since we can't compare tensor in calls, we open it and compare each elements
        decoder_mock_calls = decoder_mock.mock_calls[4:8]
        for decoder_mock_call, expected_call in zip(decoder_mock_calls, expected_calls):
            self.assertEqual(decoder_mock_call[1][0].tolist(), expected_call[1][0].tolist())  # the tensor
            self.assertEqual(decoder_mock_call[1][1], expected_call[1][1])  # The other element of the call
