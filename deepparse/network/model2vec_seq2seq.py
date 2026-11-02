# pylint: disable=too-many-arguments, too-many-positional-arguments

from .fasttext_seq2seq import FastTextSeq2SeqModel


class Model2VecSeq2SeqModel(FastTextSeq2SeqModel):
    """
    Model2Vec Seq2Seq network, the lightest of our models (in ``RAM`` consumption and speed). Like FastText, the
    words are vectorized before the network, so it shares the FastText forward pass; only the input size differs.

    Args:
        input_size (int): The input size of the encoder (i.e. the size of the Model2Vec embedding). The default value
            is ``256``.
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
        input_size: int = 256,
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
