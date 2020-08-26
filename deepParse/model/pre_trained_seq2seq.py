import os
from abc import ABC

import torch
import torch.nn as nn
from torch import load

from .decoder import Decoder
from .encoder import Encoder
from ..tools import download_weights


class PretrainedSeq2SeqModel(ABC, nn.Module):
    """
    Abstract class for callable pre trained Seq2Seq model. The model use the pre trained config for the encoder and
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

        self.decoder = Decoder(input_size=1, hidden_size=1024, num_layers=1, output_size=9)

    def _load_pre_trained_weights(self, model_type: str) -> None:
        """
        Method to resolved the loading of the pretrained weights 
        Args:
            model_type (str): The model pretrained weights to load. 
        """
        root_path = os.path.join(os.path.expanduser('~'), f".cache/deepParse")
        model_path = os.path.join(root_path, f"{model_type}.ckpt")

        if not os.path.isfile(model_path):
            download_weights(model_type, root_path)

        all_layers_params = load(model_path, map_location=self.device)

        # if model_type == "fasttext":
        #     pass
        # elif model_type == "bpemb":
        #     embedding_network = OrderedDict(
        #         [(key, value) for key, value in all_layers_params.items() if key.startswith("embedding_network")])
        #     self.embedding_network.load_state_dict(embedding_network)
        #     encoder = OrderedDict(
        #         [(key, value) for key, value in all_layers_params.items() if key.startswith("encoder")])
        #     self.encoder.load_state_dict(encoder)
        #     decoder = OrderedDict(
        #         [(key, value) for key, value in all_layers_params.items() if key.startswith("decoder")])
        #     self.decoder.load_state_dict(decoder)
        # else:
        #     pass  # raise exception

        self.load_state_dict(all_layers_params)
        print("a")

        # load weights

        pass

    def _encoder_step(self, to_predict, lenghts_tensor, batch_size):  # todo get input and output
        """
        Step of the encoder.
        
        Args:
            to_predict: 
            lenghts_tensor: 
            batch_size: 
        
        Return:
            ...
        """
        decoder_hidden = self.encoder(to_predict, lenghts_tensor)

        # -1 for BOS token
        decoder_input = torch.zeros(1, batch_size, 1).to(self.device).new_full((1, batch_size, 1), -1)
        return decoder_input, decoder_hidden

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.encoder.eval()
        self.decoder.eval()
