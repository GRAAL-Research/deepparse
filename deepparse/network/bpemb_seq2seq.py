# pylint: disable=too-many-arguments, duplicate-code, too-many-locals

from typing import List, Union

import torch

from .embedding_network import EmbeddingNetwork
from .seq2seq import Seq2SeqModel


class BPEmbSeq2SeqModel(Seq2SeqModel):
    """
    BPEmb Seq2Seq network is the best of the two proposed models but takes more ``GPU``/``CPU`` resources.

     Args:
        input_size (int): The input size of the encoder (i.e. the size of the embedding). It will also be used to
            initialize the internal embeddings network input size, hidden size and output dim. The default value is
            ``300``.
        encoder_hidden_size (int): The size of the hidden layer(s) of the encoder. The default value is ``1024``.
        encoder_num_layers (int): The number of hidden layers of the encoder. The default value is ``1``.
        decoder_hidden_size (int): The size of the hidden layer(s) of the decoder. The default value is ``1024``.
        decoder_num_layers (int): The number of hidden layers of the decoder. The default value is ``1``.
        output_size (int): The size of the prediction layers (i.e. the number of tags to predict). The default value is
            ``9``.
        attention_mechanism (bool): Either or not to use the attention mechanism. The default value is ``False``.
    """

    def __init__(
        self,
        input_size: int = 300,
        encoder_hidden_size: int = 1024,
        encoder_num_layers: int = 1,
        decoder_hidden_size: int = 1024,
        decoder_num_layers: int = 1,
        output_size: int = 9,
        attention_mechanism: bool = False,
    ) -> None:
        super().__init__(
            input_size=input_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_layers=decoder_num_layers,
            output_size=output_size,
            attention_mechanism=attention_mechanism,
        )

        self.embedding_network = EmbeddingNetwork(
            input_size=input_size, hidden_size=input_size, projection_size=input_size
        )

    def forward(
        self,
        to_predict: torch.Tensor,
        decomposition_lengths: List,
        lengths: List,
        target: Union[torch.LongTensor, None] = None,
    ) -> torch.Tensor:
        """
        Callable method as per PyTorch forward method to get tags prediction over the components of
        an address.
        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            decomposition_lengths (list) : The lengths of the decomposed words of the batch elements (since packed).
            lengths (list) : The lengths of the batch elements (since packed).
            target (~torch.LongTensor) : The target of the batch element, used only when we retrain the model since
                we do
                `teacher forcing <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/>`_.
                The default value is ``None`` since we mostly don't have the target except for retraining.
        Return:
            A Tensor of the predicted sequence.
        """
        batch_size = to_predict.size(0)

        embedded_output = self.embedding_network(to_predict, decomposition_lengths)

        decoder_input, decoder_hidden, encoder_outputs = self._encoder_step(embedded_output, lengths, batch_size)

        prediction_sequence = self._decoder_step(
            decoder_input,
            decoder_hidden,
            encoder_outputs,
            target,
            lengths,
            batch_size,
        )
        return prediction_sequence
