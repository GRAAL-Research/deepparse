from typing import Union, List

import torch

from .embedding_network import EmbeddingNetwork
from .pre_trained_seq2seq import PretrainedSeq2SeqModel


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

    def __call__(self, to_predict: torch.Tensor, lengths_tensor: torch.Tensor,
                 decomposition_lengths: List) -> torch.Tensor:
        """
            Callable method to get tags prediction over the components of an address.

            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                lengths_tensor (~torch.Tensor) : The lengths of the batch elements (since packed).
                decomposition_lengths (List) : The lengths of the decomposed words of the batch elements (since packed).

            Return:
                The tensor of the address components tags predictions.
        """
        batch_size = to_predict.size(0)

        embedded_output = self.embedding_network(to_predict, decomposition_lengths)

        decoder_input, decoder_hidden = self._encoder_step(embedded_output, lengths_tensor, batch_size)

        decoder_predict = self.decoder(decoder_input, decoder_hidden)

        return decoder_predict

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.embedding_network.eval()
        self.eval()
