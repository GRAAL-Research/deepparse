import os
from abc import ABC
from typing import Union

import torch
import torch.nn as nn
from torch import load

from deepParse.model import Decoder
from deepParse.model import Encoder
from deepParse.model.embedding_network import EmbeddingNetwork
from deepParse.tools import download_weights


class PretrainedSeq2SeqModel(ABC, nn.Module):
    """
    Abstract class for callable pre trained Seq2Seq model. The model use the pre trained config for the encoder and
    decoder:
        - Encoder: `input_size = 600`, `hidden_size = 600` and `num_layers = 1`
        - Decoder: `input_size = 1`, `hidden_size = 600`, `num_layers = 1` and `output_size = 9` (the number of tags)

     Args:
        device (str): The device tu use for the prediction, can either be a GPU or a CPU.
    """

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device

        self.encoder = Encoder(input_size=600, hidden_size=600, num_layers=1)

        self.decoder = Decoder(input_size=1, hidden_size=600, num_layers=1, output_size=9)

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


# pretrained class
class PretrainedFastTextSeq2SeqModel(PretrainedSeq2SeqModel):
    """
    FastText pre trained Seq2Seq model, the lightest of the two (in GPU/CPU consumption) for a little less accuracy.

     Args:
        device (str): The device tu use for the prediction, can either be a GPU or a CPU.
    """

    def __init__(self, device: Union[int, str]) -> None:
        super().__init__(device)

        self._load_pre_trained_weights("fasttext")

    def __call__(self, to_predict, lenghts_tensor):  # todo get input and output
        """
            Callable method to get tags prediction over the components of an address.

            Args:
                to_predict (): 
                lenghts_tensor () :

            Return:
                The address components tags predictions.
        """
        batch_size = to_predict.size(0)

        decoder_input, decoder_hidden = self._encoder_step(to_predict, lenghts_tensor, batch_size)

        decoder_predict = self.decoder(decoder_input, decoder_hidden)

        return decoder_predict


class PretrainedBPEmbSeq2SeqModel(PretrainedSeq2SeqModel):
    """
    BPEmb pre trained Seq2Seq model, the best of the two, but take the more GPU/CPU resources.

     Args:
        device (str): The device tu use for the prediction, can either be a GPU or a CPU.
    """

    def __init__(self, device: Union[int, str]):
        super().__init__(device)

        self.embedding_network = EmbeddingNetwork(input_size=300, hidden_size=300)

        self._load_pre_trained_weights("bpemb")

    def __call__(self, to_predict, lenghts_tensor, decomposition_lengths):  # todo get input and output
        """
            Callable method to get tags prediction over the components of an address.

            Args:
                to_predict (): 
                lenghts_tensor () :
                decomposition_lengths () :

            Return:
                The address components tags predictions.
        """
        batch_size = to_predict.size(0)

        embedded_output = self.embedding_network(to_predict, decomposition_lengths)

        decoder_input, decoder_hidden = self._encoder_step(embedded_output, lenghts_tensor, batch_size)

        decoder_predict = self.decoder(decoder_input, decoder_hidden)

        return decoder_predict

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.embedding_network.eval()
        self.eval()
