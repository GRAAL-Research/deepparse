import random
from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn

from deepParse.model import Decoder
from deepParse.model import Encoder
from deepParse.model.embedding_network import EmbeddingNetwork


class PretrainedSeq2SeqModel(ABC, nn.Module):
    """
    Abstract class for callable #todo .
    """

    def __init__(self, device: Union[int, str], batch_size: Union[int, None] = None) -> None:
        super().__init__()
        self.device = "cuda:%d" % str(device)
        self.batch_size = 1 if batch_size is None else batch_size

        self.output_size = 1  # todo use pre trained size

        self.encoder = Encoder(1, 1, 1, self.batch_size)
        self.encoder.to(device)

        self.decoder = Decoder(1, 1, 1, 1)
        self.decoder.to(device)

    @abstractmethod
    def __call__(self, to_predict, lenghts_tensor):
        pass

    def _get_dynamic_batch(self, input_):
        return 1 if input_.size(0) < self.batch_size else self.batch_size


# pretrained class
class FastTextAddressSeq2SeqModel(PretrainedSeq2SeqModel):
    def __init__(self, device: Union[int, str], batch_size: Union[int, None] = None) -> None:
        super().__init__(device, batch_size)

    def __call__(self, to_predict, lenghts_tensor):
        batch_size = self._get_dynamic_batch(to_predict)

        decoder_input, decoder_hidden = self._encoder_step(to_predict, lenghts_tensor, batch_size)

        decoder_output, _ = self.decoder(decoder_input, decoder_hidden)

        return decoder_output

    def _encoder_step(self, input_, lenghts_tensor, batch_size):
        decoder_hidden = self.encoder(input_, lenghts_tensor)

        # -1 for BOS token
        decoder_input = torch.zeros(1, batch_size, 1).cuda(self.device).new_full((1, batch_size, 1), -1)
        return decoder_input, decoder_hidden


class BPEmbAddressSeq2SeqModel(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, encoder_hidden_size, num_encoding_layers,
                 decoder_hidden_size, num_decoding_layers, output_size, batch_size, EOS_token, device):
        super().__init__()
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        self.embedding_network = EmbeddingNetwork(embedding_input_size, embedding_hidden_size,
                                                  embedding_projection_size, self.batch_size, device)
        self.embedding_network.to(device)

        self.encoder = Encoder(encoder_input_size, encoder_hidden_size, num_encoding_layers, self.batch_size, device)
        self.encoder.to(device)

        self.decoder = Decoder(decoder_input_size, decoder_hidden_size, num_decoding_layers, output_size,
                               self.batch_size, device)
        self.decoder.to(device)

        self.EOS_token = EOS_token

    def forward(self, input_, decomposition_lengths, lenghts_tensor, target=None):
        embedded_input = self.embedding_network(input_.to(self.device), decomposition_lengths)

        if embedded_input.size(0) < self.batch_size:
            batch_size = 1
        else:
            batch_size = self.batch_size

        encoder_hidden = self.encoder(embedded_input.to(self.device), lenghts_tensor)

        decoder_hidden = encoder_hidden
        max_length = lenghts_tensor[0].item()

        output_sequence = torch.zeros(max_length + 1, batch_size, self.output_size).to(self.device)

        BOS_token = -1
        decoder_input = torch.zeros(1, batch_size, 1).to(self.device).new_full((1, batch_size, 1), BOS_token)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        output_sequence[0] = decoder_output

        _, decoder_input = decoder_output.topk(1)

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

                _, decoder_input = decoder_output.topk(1)

        return output_sequence
