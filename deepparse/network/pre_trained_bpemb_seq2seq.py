from typing import List

import torch

from .embedding_network import EmbeddingNetwork
from .pre_trained_seq2seq import PreTrainedSeq2SeqModel


class PreTrainedBPEmbSeq2SeqModel(PreTrainedSeq2SeqModel):
    """
    BPEmb pre-trained Seq2Seq network, the best of the two, but takes more ``GPU``/``CPU`` resources.

     Args:
        device (~torch.device): The device tu use for the prediction.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

        # pre-trained params (the 300)
        self.embedding_network = EmbeddingNetwork(input_size=300, hidden_size=300, projection_size=300)
        self.embedding_network.to(self.device)

        self._load_pre_trained_weights("bpemb")

    def forward(self, to_predict: torch.Tensor, decomposition_lengths: List,
                lengths_tensor: torch.Tensor) -> torch.Tensor:
        """
            Callable method as per PyTorch forward method to get tags prediction over the components of
            an address.
            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                decomposition_lengths (list) : The lengths of the decomposed words of the batch elements (since packed).
                lengths_tensor (~torch.Tensor) : The lengths of the batch elements (since packed).

            Return:
                The tensor of the address components tags predictions.
        """
        batch_size = to_predict.size(0)

        embedded_output = self.embedding_network(to_predict, decomposition_lengths)

        decoder_input, decoder_hidden = self._encoder_step(embedded_output, lengths_tensor, batch_size)

        max_length = lengths_tensor[0].item()
        prediction_sequence = self._decoder_steps(decoder_input, decoder_hidden, max_length, batch_size)

        return prediction_sequence
