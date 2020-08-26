from typing import Union

from .pre_trained_seq2seq import PretrainedSeq2SeqModel


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
