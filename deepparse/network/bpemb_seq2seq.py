from typing import List, Union

import torch

from .embedding_network import EmbeddingNetwork
from .seq2seq import Seq2SeqModel


class BPEmbSeq2SeqModel(Seq2SeqModel):
    """
    BPEmb Seq2Seq network, the best of the two model we propose, but takes more ``GPU``/``CPU`` resources.

     Args:
        device (~torch.device): The device tu use for the prediction.
        verbose (bool): Turn on/off the verbosity of the model. The default value is True.
        path_to_retrained_model (Union[str, None]): The path to the retrained model to use for the seq2seq.
    """

    def __init__(self,
                 device: torch.device,
                 verbose: bool = True,
                 path_to_retrained_model: Union[str, None] = None) -> None:
        super().__init__(device, verbose)

        # design dimension params (the 300)
        self.embedding_network = EmbeddingNetwork(input_size=300, hidden_size=300, projection_size=300)
        self.embedding_network.to(self.device)

        if path_to_retrained_model is not None:
            self._load_weights(path_to_retrained_model)
        else:
            self._load_pre_trained_weights("bpemb")

    def forward(self,
                to_predict: torch.Tensor,
                decomposition_lengths: List,
                lengths_tensor: torch.Tensor,
                target: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Callable method as per PyTorch forward method to get tags prediction over the components of
        an address.
        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            decomposition_lengths (list) : The lengths of the decomposed words of the batch elements (since packed).
            lengths_tensor (~torch.Tensor) : The lengths of the batch elements (since packed).
            target (~torch.Tensor) : The target of the batch element, use only when we retrain the model since we do
                `teacher forcing <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/>`_.
                Default value is None since we mostly don't have the target except for retrain.
        Return:
            The tensor of the address components tags predictions.
        """
        batch_size = to_predict.size(0)

        embedded_output = self.embedding_network(to_predict, decomposition_lengths)

        decoder_input, decoder_hidden = self._encoder_step(embedded_output, lengths_tensor, batch_size)

        max_length = lengths_tensor.max().item()
        prediction_sequence = self._decoder_step(decoder_input, decoder_hidden, target, max_length, batch_size)

        return prediction_sequence
