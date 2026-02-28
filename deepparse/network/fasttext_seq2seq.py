# pylint: disable=too-many-arguments, duplicate-code, too-many-locals

from typing import List, Union

import torch

from .seq2seq import Seq2SeqModel


class FastTextSeq2SeqModel(Seq2SeqModel):
    """
    FastText Seq2Seq network, the lightest of the two models we propose (in ``GPU``/``CPU`` consumption) for a little
    less accuracy.

    Args:
        input_size (int): The input size of the encoder (i.e. the size of the embedding). The default value is ``300``.
        encoder_hidden_size (int): The size of the encoder's hidden layer(s). The default value is ``1024``.
        encoder_num_layers (int): The number of hidden layers of the encoder. The default value is ``1``.
        decoder_hidden_size (int): The size of the decoder's hidden layer(s). The default value is ``1024``.
        decoder_num_layers (int): The number of hidden layers of the decoder. The default value is ``1``.
        output_size (int): The size of the prediction layers (i.e. the number of tags to predict). The default value
            is ``9``.
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

    def forward(
        self,
        to_predict: torch.Tensor,
        lengths: List,
        target: Union[torch.LongTensor, None] = None,
    ) -> torch.Tensor:
        """
        Callable method as per PyTorch forward method to get tags prediction over the components of
        an address.

        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            lengths (list) : The lengths of the batch elements (since packed).
            target (~torch.LongTensor) : The target of the batch element, use only when we retrain the model since we do
                `teacher forcing <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/>`_.
                The default value is ``None`` since we mostly don't have the target except for retrain.

        Return:
            A Tensor of the predicted sequence.
        """
        batch_size = to_predict.size(0)

        decoder_input, decoder_hidden, encoder_outputs = self._encoder_step(to_predict, lengths, batch_size)

        prediction_sequence = self._decoder_step(
            decoder_input,
            decoder_hidden,
            encoder_outputs,
            target,
            lengths,
            batch_size,
        )

        return prediction_sequence
