# pylint: disable=too-many-arguments
import os
import random
import warnings
from abc import ABC
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from ..tools import CACHE_PATH
from ..tools import download_weights, latest_version


class Seq2SeqModel(ABC, nn.Module):
    """
    Abstract class for Seq2Seq networks.

     Args:
        device (~torch.device): The device tu use for the prediction.
        input_size (int): The input size of the encoder (i.e. the embeddings size). The default value is 300.
        encoder_hidden_size (int): The size of the hidden layer(s) of the encoder. The default value is 1024.
        encoder_num_layers (int): The number of hidden layers of the encoder. The default value is 1.
        decoder_hidden_size (int): The size of the hidden layer(s) of the decoder. The default value is 1024.
        decoder_num_layers (int): The number of hidden layers of the decoder. The default value is 1.
        output_size (int): The size of the prediction layers (i.e. the number of tag to predict).
        attention_mechanism (bool): Either or not to use attention mechanism. The default value is False.
        verbose (bool): Turn on/off the verbosity of the model. The default value is True.
    """

    def __init__(self,
                 device: torch.device,
                 input_size: int,
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 output_size: int,
                 attention_mechanism: bool = False,
                 verbose: bool = True) -> None:
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.attention_mechanism = attention_mechanism

        self.encoder = Encoder(input_size=input_size, hidden_size=encoder_hidden_size, num_layers=encoder_num_layers)
        self.encoder.to(self.device)

        self.decoder = Decoder(input_size=encoder_num_layers,
                               hidden_size=decoder_hidden_size,
                               num_layers=decoder_num_layers,
                               output_size=output_size,
                               attention_mechanism=self.attention_mechanism)

        self.decoder.to(self.device)

        self.output_size = output_size

    def same_output_dim(self, size: int) -> bool:
        """
        Verify if the output dimension is similar to ``size``.

        Args:
            size (int): The dimension size to compare the output dim to.

        Return: A bool, True if output dim is equal to ``size``, False otherwise.
        """
        return size == self.output_size

    def handle_new_output_dim(self, new_dim: int) -> None:
        """
        Update the new output dimension
        """
        self.decoder.linear_layer_set_up(output_size=new_dim)
        self.output_size = new_dim

    def _load_pre_trained_weights(self, model_type: str) -> None:
        """
        Method to download and resolved the loading (into the network) of the pre-trained weights.

        Args:
            model_type (str): The network pre-trained weights to load.
        """
        model_path = os.path.join(CACHE_PATH, f"{model_type}.ckpt")

        if not os.path.isfile(model_path):
            download_weights(model_type, CACHE_PATH, verbose=self.verbose)
        elif not latest_version(model_type, cache_path=CACHE_PATH):
            if self.verbose:
                warnings.warn("A new version of the pre-trained model is available. "
                              "The newest model will be downloaded.")
            download_weights(model_type, CACHE_PATH, verbose=self.verbose)

        all_layers_params = torch.load(model_path, map_location=self.device)
        self.load_state_dict(all_layers_params)

    def _load_weights(self, path_to_retrained_model: str) -> None:
        """
        Method to load (into the network) the weights.

        Args:
            path_to_retrained_model (str): The path to the fine-tuned model.
        """
        all_layers_params = torch.load(path_to_retrained_model, map_location=self.device)
        if isinstance(all_layers_params, dict) and not isinstance(all_layers_params, OrderedDict):
            # Case where we have a retrained model with a different tagging space
            all_layers_params = all_layers_params.get("address_tagger_model")
        self.load_state_dict(all_layers_params)

    def _encoder_step(self, to_predict: torch.Tensor, lengths_tensor: torch.Tensor, batch_size: int) -> Tuple:
        """
        Step of the encoder.

        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            lengths_tensor (~torch.Tensor): The lengths of the batch elements (since packed).
            batch_size (int): The number of element in the batch.

        Return:
            A tuple (``x``, ``y``, ``z``) where ``x`` is the decoder input (a zeros tensor), ``y`` is the decoder
            hidden states and ``z`` is the encoder outputs for the attention weighs if needed.
        """
        encoder_outputs, decoder_hidden = self.encoder(to_predict, lengths_tensor)

        # -1 for BOS token
        decoder_input = torch.zeros(1, batch_size, 1).to(self.device).new_full((1, batch_size, 1), -1)

        return decoder_input, decoder_hidden, encoder_outputs

    def _decoder_step(self, decoder_input: torch.Tensor, decoder_hidden: tuple, encoder_outputs: torch.Tensor,
                      target: Union[torch.Tensor, None], lengths_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Step of the encoder.

        Args:
            decoder_input (~torch.Tensor): The decoder input (so the encode output).
            decoder_hidden (~torch.Tensor): The encoder hidden state (so the encode hidden state).
            encoder_outputs (~torch.Tensor): The encoder outputs for the attention mechanism weighs if needed.
            target (~torch.Tensor) : The target of the batch element, use only when we retrain the model since we do
                `teacher forcing <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/>`_.
            lengths_tensor (~torch.Tensor): The lengths of the batch elements (since packed).
            batch_size (int): Number of element in the batch.

        Return:
            A Tensor of the predicted sequence.
        """
        max_length = lengths_tensor.max().item()

        # The empty prediction sequence
        # +1 for the EOS
        prediction_sequence = torch.zeros(max_length + 1, batch_size, self.output_size).to(self.device)

        # We decode the first token
        decoder_output, decoder_hidden, attention_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                         lengths_tensor)

        if attention_weights is not None:
            # We fill the attention
            attention_output = torch.ones(max_length + 1, batch_size, 1, max_length)
            attention_output[0] = attention_weights

        # We fill the first token prediction
        prediction_sequence[0] = decoder_output

        # The decoder next step input (the predicted idx of the previous token)
        _, decoder_input = decoder_output.topk(1)

        # we loop the same steps for the rest of the sequence
        if target is not None and random.random() < 0.5:
            # force the real target value instead of the predicted one to help learning
            target = target.transpose(0, 1)
            for idx in range(max_length):
                decoder_input = target[idx].view(1, batch_size, 1)
                decoder_output, decoder_hidden, attention_weights = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs, lengths_tensor)
                prediction_sequence[idx + 1] = decoder_output

        else:
            for idx in range(max_length):
                decoder_output, decoder_hidden, attention_weights = self.decoder(decoder_input.view(1, batch_size,
                                                                                                    1), decoder_hidden,
                                                                                 encoder_outputs, lengths_tensor)

                prediction_sequence[idx + 1] = decoder_output

                _, decoder_input = decoder_output.topk(1)

        return prediction_sequence
