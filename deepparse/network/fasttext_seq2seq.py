from typing import Union

import torch

from .seq2seq import Seq2SeqModel


class FastTextSeq2SeqModel(Seq2SeqModel):
    """
    FastText Seq2Seq network, the lightest of the two model we propose (in ``GPU``/``CPU`` consumption) for a little
    less accuracy.

    Args:
        device (~torch.device): The device tu use for the prediction.
        output_size (int): The size of the prediction layers (i.e. the number of tag to predict).
        verbose (bool): Turn on/off the verbosity of the model. The default value is True.
        path_to_retrained_model (Union[str, None]): The path to the retrained model to use for the seq2seq.
    """

    def __init__(self,
                 device: torch.device,
                 output_size: int,
                 verbose: bool = True,
                 path_to_retrained_model: Union[str, None] = None) -> None:
        super().__init__(device, output_size, verbose)

        if path_to_retrained_model is not None:
            self._load_weights(path_to_retrained_model)
        else:
            self._load_pre_trained_weights("fasttext")

    def forward(self,
                to_predict: torch.Tensor,
                lengths_tensor: torch.Tensor,
                target: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Callable method as per PyTorch forward method to get tags prediction over the components of
        an address.

        Args:
            to_predict (~torch.Tensor): The elements to predict the tags.
            lengths_tensor (~torch.Tensor) : The lengths of the batch elements (since packed).
            target (~torch.Tensor) : The target of the batch element, use only when we retrain the model since we do
                `teacher forcing <https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/>`_.
                Default value is None since we mostly don't have the target except for retrain.

        Return:
            The tensor of the address components tags predictions.
        """
        batch_size = to_predict.size(0)

        decoder_input, decoder_hidden = self._encoder_step(to_predict, lengths_tensor, batch_size)

        max_length = lengths_tensor.max().item()
        decoder_predict = self._decoder_step(decoder_input, decoder_hidden, target, max_length, batch_size)

        return decoder_predict
