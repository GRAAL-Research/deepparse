import torch

from .pre_trained_seq2seq import PreTrainedSeq2SeqModel


class PreTrainedFastTextSeq2SeqModel(PreTrainedSeq2SeqModel):
    """
    FastText pre-trained Seq2Seq network, the lightest of the two (in ``GPU``/``CPU`` consumption) for a little less
    accuracy.

     Args:
        device (~torch.device): The device tu use for the prediction.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

        self._load_pre_trained_weights("fasttext")

    def forward(self, to_predict: torch.Tensor, lengths_tensor: torch.Tensor) -> torch.Tensor:
        """
            Callable method as per PyTorch forward method to get tags prediction over the components of
            an address.

            Args:
                to_predict (~torch.Tensor): The elements to predict the tags.
                lengths_tensor (~torch.Tensor) : The lengths of the batch elements (since packed).

            Return:
                The tensor of the address components tags predictions.
        """
        batch_size = to_predict.size(0)

        decoder_input, decoder_hidden = self._encoder_step(to_predict, lengths_tensor, batch_size)

        max_length = lengths_tensor[0].item()
        decoder_predict = self._decoder_steps(decoder_input, decoder_hidden, max_length, batch_size)

        return decoder_predict
