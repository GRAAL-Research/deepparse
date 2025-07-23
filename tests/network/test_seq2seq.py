# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# Since we use patch we skip the unused argument error
# We also skip protected-access since we test the _load_weights
# pylint: disable=protected-access, unused-argument, not-callable

# Pylint kick for temporary directory
# pylint: disable=consider-using-with

import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest import skipIf

from deepparse.network import Seq2SeqModel
from tests.tools import create_file


class Seq2SeqTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # cls.a_torch_device = torch.device("cuda:0")
        # cls.a_cpu_device = torch.device("cpu")

        cls.encoder_input_size_dim = 300
        cls.encoder_hidden_size = 1024
        cls.encoder_num_layers = 1
        cls.decoder_input_size_dim = 1
        cls.decoder_hidden_size = 1024
        cls.decoder_num_layers = 1
        cls.decoder_output_size = 9

        cls.a_fake_retrain_path = "a/fake/path/retrain/model"

        cls.temp_dir_obj = TemporaryDirectory()
        cls.fake_cache_dir = os.path.join(cls.temp_dir_obj.name, "fake_cache")
        os.makedirs(cls.fake_cache_dir, exist_ok=True)

        cls.a_hash_value = "f67a0517c70a314bdde0b8440f21139d"

        version_file_path = os.path.join(cls.fake_cache_dir, "fasttext.version")
        create_file(version_file_path, cls.a_hash_value)

        cls.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

        cls.a_model_type = "a_model_type"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        seq2seq_model = Seq2SeqModel(
            input_size=self.encoder_input_size_dim,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_num_layers=self.encoder_num_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_layers=self.decoder_num_layers,
            output_size=self.decoder_output_size,
        )

        self.assertEqual(self.encoder_input_size_dim, seq2seq_model.encoder.lstm.input_size)
        self.assertEqual(self.encoder_hidden_size, seq2seq_model.encoder.lstm.hidden_size)
        self.assertEqual(self.encoder_num_layers, seq2seq_model.encoder.lstm.num_layers)
        # self.assertEqual(self.a_cpu_device, seq2seq_model.encoder.lstm.all_weights[0][0].device)

        self.assertEqual(self.decoder_input_size_dim, seq2seq_model.decoder.lstm.input_size)
        self.assertEqual(self.decoder_hidden_size, seq2seq_model.decoder.lstm.hidden_size)
        self.assertEqual(self.decoder_num_layers, seq2seq_model.decoder.lstm.num_layers)
        self.assertEqual(self.decoder_output_size, seq2seq_model.decoder.linear.out_features)
        # self.assertEqual(self.a_cpu_device, seq2seq_model.decoder.lstm.all_weights[0][0].device)

    @skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
    def test_whenInstantiateASeq2SeqModelGPU_thenParametersAreOk(self):
        seq2seq_model = Seq2SeqModel(
            input_size=self.encoder_input_size_dim,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_num_layers=self.encoder_num_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_layers=self.decoder_num_layers,
            output_size=self.decoder_output_size,
        )

        self.assertEqual(self.encoder_input_size_dim, seq2seq_model.encoder.lstm.input_size)
        self.assertEqual(self.encoder_hidden_size, seq2seq_model.encoder.lstm.hidden_size)
        self.assertEqual(self.encoder_num_layers, seq2seq_model.encoder.lstm.num_layers)
        # self.assertEqual(self.a_torch_device, seq2seq_model.encoder.lstm.all_weights[0][0].device)

        self.assertEqual(self.decoder_input_size_dim, seq2seq_model.decoder.lstm.input_size)
        self.assertEqual(self.decoder_hidden_size, seq2seq_model.decoder.lstm.hidden_size)
        self.assertEqual(self.decoder_num_layers, seq2seq_model.decoder.lstm.num_layers)
        self.assertEqual(self.decoder_output_size, seq2seq_model.decoder.linear.out_features)
        # self.assertEqual(self.a_torch_device, seq2seq_model.decoder.lstm.all_weights[0][0].device)

    def test_whenSameOutput_thenReturnTrue(self):
        seq2seq_model = Seq2SeqModel(
            input_size=self.encoder_input_size_dim,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_num_layers=self.encoder_num_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_layers=self.decoder_num_layers,
            output_size=self.decoder_output_size,
        )
        self.assertTrue(seq2seq_model.same_output_dim(self.decoder_output_size))

    def test_whenNotSameOutput_thenReturnFalse(self):
        seq2seq_model = Seq2SeqModel(
            input_size=self.encoder_input_size_dim,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_num_layers=self.encoder_num_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_layers=self.decoder_num_layers,
            output_size=self.decoder_output_size,
        )
        self.assertFalse(seq2seq_model.same_output_dim(self.decoder_output_size - 1))

    def test_whenHandleNewOutputDim_thenProperlyHandleNewDim(self):
        seq2seq_model = Seq2SeqModel(
            input_size=self.encoder_input_size_dim,
            encoder_hidden_size=self.encoder_hidden_size,
            encoder_num_layers=self.encoder_num_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_layers=self.decoder_num_layers,
            output_size=self.decoder_output_size,
        )

        a_new_dim = 1
        seq2seq_model.handle_new_output_dim(a_new_dim)

        expected = a_new_dim
        actual = seq2seq_model.output_size
        self.assertEqual(expected, actual)

        actual = seq2seq_model.decoder.linear.out_features
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
