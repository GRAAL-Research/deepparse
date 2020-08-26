import os
from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
from torch import load

from .decoder import Decoder
from .encoder import Encoder
from ..tools import download_weights


class PreTrainedSeq2SeqModel(ABC, nn.Module):
    """
    Abstract class for callable pre-trained Seq2Seq model. The model use the pre-trained config for the encoder and
    decoder:
        - Encoder: `input_size = 300`, `hidden_size = 1024` and `num_layers = 1`
        - Decoder: `input_size = 1`, `hidden_size = 1024`, `num_layers = 1` and `output_size = 9` (the number of tags)

     Args:
        device (str): The device tu use for the prediction, can either be a GPU or a CPU.
    """

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device

        self.encoder = Encoder(input_size=300, hidden_size=1024, num_layers=1)
        self.encoder.to(self.device)

        self.decoder = Decoder(input_size=1, hidden_size=1024, num_layers=1, output_size=9)
        self.decoder.to(self.device)

    def _load_pre_trained_weights(self, model_type: str) -> None:
        """
        Method to download and resolved the loading (into the model) of the pre-trained weights.

        Args:
            model_type (str): The model pre-trained weights to load.
        """
        root_path = os.path.join(os.path.expanduser('~'), ".cache/deepParse")
        model_path = os.path.join(root_path, f"{model_type}.ckpt")

        if not os.path.isfile(model_path):
            download_weights(model_type, root_path)

        all_layers_params = load(model_path, map_location=self.device)

        self.load_state_dict(all_layers_params)

    def _encoder_step(self, to_predict: torch.Tensor, lengths_tensor: torch.Tensor, batch_size: int) -> Tuple:
        """
        Step of the encoder.

        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            lengths_tensor (~torch.Tensor): The lengths of the batch elements (since packed).
            batch_size (int): The number of element in the batch.

        Return:
            A tuple (x, y) where x is the decoder input (a zeros tensor) and y is the decoder hidden states.
        """
        decoder_hidden = self.encoder(to_predict, lengths_tensor)

        # -1 for BOS token
        decoder_input = torch.zeros(1, batch_size, 1).to(self.device).new_full((1, batch_size, 1), -1)
        return decoder_input, decoder_hidden

    def _decoder_steps(self, decoder_input: torch.Tensor, decoder_hidden: torch.Tensor, max_length: int,
                       batch_size: int) -> torch.Tensor:
        # The empty prediction sequence
        # +1 for the EOS
        # 9 for the output size (9 tokens)
        prediction_sequence = torch.zeros(max_length + 1, batch_size, 9).cuda(self.device)

        # we decode the first token
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        # we fill the first token prediction
        prediction_sequence[0] = decoder_output

        # the decoder next step input (the predicted idx of the previous token)
        _, decoder_input = decoder_output.topk(1)

        # we loop the same steps for the rest of the sequence
        for idx in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input.view(1, batch_size, 1), decoder_hidden)

            prediction_sequence[idx + 1] = decoder_output

            _, decoder_input = decoder_output.topk(1)

        return prediction_sequence  # the sequence is now fully parse

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.encoder.eval()
        self.decoder.eval()
