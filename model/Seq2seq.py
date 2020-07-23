import torch
import torch.nn as nn
from model.Decoder import Decoder
from model.Encoder import Encoder
from model.EmbeddingNetwork import EmbeddingNetwork
import random


class Seq2seq(nn.Module):
    def __init__(self, embedding_input_size, encoder_input_size, decoder_input_size, embedding_hidden_size, embedding_projection_size, encoder_hidden_size, num_encoding_layers, 
                    decoder_hidden_size, num_decoding_layers, output_size, batch_size, EOS_token, device):
        super().__init__()
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        self.embedding_network = EmbeddingNetwork(embedding_input_size, embedding_hidden_size, embedding_projection_size, self.batch_size, device)
        self.embedding_network.cuda(device)

        self.encoder = Encoder(encoder_input_size, encoder_hidden_size, num_encoding_layers, self.batch_size, device)
        self.encoder.cuda(device)

        self.decoder = Decoder(decoder_input_size, decoder_hidden_size, num_decoding_layers, output_size, self.batch_size, device)
        self.decoder.cuda(device)

        self.EOS_token = EOS_token

    def forward(self, input_, decomposition_lengths, lenghts_tensor, target=None):
        embedded_input = self.embedding_network(input_.cuda(self.device), decomposition_lengths)

        if embedded_input.size(0) < self.batch_size:
            batch_size = 1
        else:
            batch_size = self.batch_size

        encoder_hidden = self.encoder(embedded_input.cuda(self.device), lenghts_tensor)

        decoder_hidden = encoder_hidden
        max_length = lenghts_tensor[0].item()

        output_sequence = torch.zeros(max_length + 1, batch_size, self.output_size).cuda(self.device)

        BOS_token = -1
        decoder_input = torch.zeros(1, batch_size, 1).cuda(self.device).new_full((1, batch_size, 1), BOS_token)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        
        output_sequence[0] = decoder_output

        _ , decoder_input = decoder_output.topk(1)

        if not target is None and random.random() < 0.5:
            target = target.transpose(0, 1)
            for idx in range(max_length):
                decoder_input = target[idx].view(1, batch_size, 1)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                output_sequence[idx + 1] = decoder_output

        else:
            for idx in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input.view(1, batch_size, 1), decoder_hidden)

                output_sequence[idx + 1] = decoder_output

                _ , decoder_input = decoder_output.topk(1)

        return output_sequence
