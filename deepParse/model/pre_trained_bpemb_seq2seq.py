from typing import Union

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

    def __call__(self, to_predict, lenghts_tensor, decomposition_lengths):  # todo get input and output
        """
            Callable method to get tags prediction over the components of an address.

            Args:
                to_predict ():
                lenghts_tensor () :
                decomposition_lengths () :

            Return:
                The address components tags predictions.
        """
        batch_size = to_predict.size(0)

        embedded_output = self.embedding_network(to_predict, decomposition_lengths)

        decoder_input, decoder_hidden = self._encoder_step(embedded_output, lenghts_tensor, batch_size)

        decoder_predict = self.decoder(decoder_input, decoder_hidden)

        return decoder_predict

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.embedding_network.eval()
        self.eval()
