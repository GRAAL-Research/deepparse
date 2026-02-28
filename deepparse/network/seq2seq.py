# pylint: disable=too-many-arguments
import random
from abc import ABC
from typing import List, Tuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from ..network.decoder import Decoder
from ..network.encoder import Encoder


class Seq2SeqModel(ABC, nn.Module, PyTorchModelHubMixin):
    """
    Abstract class for Seq2Seq networks.

     Args:
        input_size (int): The input size of the encoder (i.e. the size of the embedding). The default value is ``300``.
        encoder_hidden_size (int): The size of the encoder's hidden layer(s). The default value is ``1024``.
        encoder_num_layers (int): The number of hidden layers of the encoder. The default value is ``1``.
        decoder_hidden_size (int): The size of the decoder's hidden layer(s). The default value is ``1024``.
        decoder_num_layers (int): The number of hidden layers of the decoder. The default value is ``1``.
        output_size (int): The size of the prediction layers (i.e. the number of tags to predict). The default value is
            ``9``.
        attention_mechanism (bool): Either or not to use the attention mechanism. The default value is ``False``.
    """

    def __init__(
        self,
        input_size: int,
        encoder_hidden_size: int,
        encoder_num_layers: int,
        decoder_hidden_size: int,
        decoder_num_layers: int,
        output_size: int,
        attention_mechanism: bool = False,
    ) -> None:
        super().__init__()
        self.attention_mechanism = attention_mechanism

        self.device = torch.device("cpu")

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
        )

        self.decoder = Decoder(
            input_size=encoder_num_layers,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            output_size=output_size,
            attention_mechanism=self.attention_mechanism,
        )

        self.output_size = output_size

    def to_device(self, device: torch.device):
        self.device = device

        self.to(self.device)

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
        Update the new output dimension.
        """
        self.decoder.linear_layer_set_up(output_size=new_dim)
        self.output_size = new_dim

    def _encoder_step(self, to_predict: torch.Tensor, lengths: List, batch_size: int) -> Tuple:
        """
        Step of the encoder.

        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            lengths (list): The lengths of the batch elements (since packed).
            batch_size (int): The number of elements in the batch.

        Return:
            A tuple (``x``, ``y``, ``z``) where ``x`` is the decoder input (a zeros tensor), ``y`` is the decoder
            hidden states, and ``z`` is the encoder output for the attention weighs if needed.
        """
        encoder_outputs, decoder_hidden = self.encoder(to_predict, lengths)

        # -1 for BOS token
        decoder_input = torch.zeros(1, batch_size, 1, device=self.device).new_full((1, batch_size, 1), -1)

        return decoder_input, decoder_hidden, encoder_outputs

    def _decoder_step(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: tuple,
        encoder_outputs: torch.Tensor,
        target: torch.LongTensor | None,
        lengths: List,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Step of the encoder.

        Args:
            decoder_input (~torch.Tensor): The decoder input (so the encode output).
            decoder_hidden (~torch.Tensor): The encoder's hidden state (so the encode hidden state).
            encoder_outputs (~torch.Tensor): The encoder outputs for the attention mechanism weighs if needed.
            target (~torch.LongTensor) : The target of the batch element, used only when we retrain the model since
                we do
                `teacher forcing <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/>`_.
            lengths (list): The lengths of the batch elements (since packed).
            batch_size (int): Number of elements in the batch.

        Return:
            A Tensor of the predicted sequence.
        """
        longest_sequence_length = max(lengths)

        # The empty prediction sequence.
        # +1 for the EOS.
        prediction_sequence = torch.zeros(longest_sequence_length + 1, batch_size, self.output_size, device=self.device)

        # We decode the first token.
        decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs, lengths)

        # We fill the first token prediction.
        prediction_sequence[0] = decoder_output

        # The decoder's next step input (the predicted idx of the previous token).
        _, decoder_input = decoder_output.topk(1)

        # We loop the same steps for the rest of the sequence.
        if target is not None and random.random() < 0.5:
            # Force the real target value instead of the predicted one to help learning.
            target = target.transpose(0, 1)
            for idx in range(longest_sequence_length):
                decoder_input = target[idx].view(1, batch_size, 1)
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, lengths
                )
                prediction_sequence[idx + 1] = decoder_output

        else:
            for idx in range(longest_sequence_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input.view(1, batch_size, 1),
                    decoder_hidden,
                    encoder_outputs,
                    lengths,
                )

                prediction_sequence[idx + 1] = decoder_output

                _, decoder_input = decoder_output.topk(1)

        return prediction_sequence
