from unittest import TestCase

import torch

from deepparse.network import Seq2SeqModel


class Seq2SeqTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_torch_device = torch.device("cpu")

        cls.encoder_input_size_dim = 300
        cls.encoder_hidden_size = 1024
        cls.encoder_num_layers = 1
        cls.decoder_input_size_dim = 1
        cls.decoder_hidden_size = 1024
        cls.decoder_num_layers = 1
        cls.decoder_output_size = 9

    def setUp(self) -> None:
        self.seq2seq_model = Seq2SeqModel(self.a_torch_device)

    def test_whenInstantiateASeq2SeqModel_thenParametersAreOk(self):
        self.assertEqual(self.a_torch_device, self.seq2seq_model.device)

        self.assertEqual(self.encoder_input_size_dim, self.seq2seq_model.encoder.lstm.input_size)
        self.assertEqual(self.encoder_hidden_size, self.seq2seq_model.encoder.lstm.hidden_size)
        self.assertEqual(self.encoder_num_layers, self.seq2seq_model.encoder.lstm.num_layers)
        self.assertEqual(self.a_torch_device, self.seq2seq_model.encoder.lstm.all_weights[0][0].device)

        self.assertEqual(self.decoder_input_size_dim, self.seq2seq_model.decoder.lstm.input_size)
        self.assertEqual(self.decoder_hidden_size, self.seq2seq_model.decoder.lstm.hidden_size)
        self.assertEqual(self.decoder_num_layers, self.seq2seq_model.decoder.lstm.num_layers)
        self.assertEqual(self.decoder_output_size, self.seq2seq_model.decoder.linear.out_features)
        self.assertEqual(self.a_torch_device, self.seq2seq_model.decoder.lstm.all_weights[0][0].device)
