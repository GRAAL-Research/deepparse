# pylint: disable=too-many-arguments, duplicate-code, too-many-locals

from typing import Union, List

import torch

from .seq2seq import Seq2SeqModel


class FastTextSeq2SeqModel(Seq2SeqModel):
    """
    FastText Seq2Seq network, the lightest of the two model we propose (in ``GPU``/``CPU`` consumption) for a little
    less accuracy.

    Args:
        cache_dir (str): The path to the cached directory to use for downloading (and loading) the
            model weights.
        device (~torch.device): The device tu use for the prediction.
        input_size (int): The input size of the encoder (i.e. the embeddings size). The default value is 300.
        encoder_hidden_size (int): The size of the hidden layer(s) of the encoder. The default value is 1024.
        encoder_num_layers (int): The number of hidden layers of the encoder. The default value is 1.
        decoder_hidden_size (int): The size of the hidden layer(s) of the decoder. The default value is 1024.
        decoder_num_layers (int): The number of hidden layers of the decoder. The default value is 1.
        output_size (int): The size of the prediction layers (i.e. the number of tag to predict).
        attention_mechanism (bool): Either or not to use attention mechanism. The default value is False.
        verbose (bool): Turn on/off the verbosity of the model. The default value is True.
        path_to_retrained_model (Union[str, None]): The path to the retrained model to use for the seq2seq.
    """

    def __init__(
        self,
        cache_dir: str,
        device: torch.device,
        input_size: int = 300,
        encoder_hidden_size: int = 1024,
        encoder_num_layers: int = 1,
        decoder_hidden_size: int = 1024,
        decoder_num_layers: int = 1,
        output_size: int = 9,
        attention_mechanism: bool = False,
        verbose: bool = True,
        path_to_retrained_model: Union[str, None] = None,
        pre_trained_weights: bool = True,
        offline: bool = False,
    ) -> None:
        super().__init__(
            device,
            input_size=input_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_layers=decoder_num_layers,
            output_size=output_size,
            attention_mechanism=attention_mechanism,
            verbose=verbose,
        )

        model_weights_name = "fasttext"
        if attention_mechanism:
            model_weights_name += "_attention"

        if path_to_retrained_model is not None:
            self._load_weights(path_to_retrained_model)

            version = "FineTunedModel" + self._load_version(model_type=model_weights_name, cache_dir=cache_dir)
        elif pre_trained_weights:
            # Means we use the pretrained weights
            self._load_pre_trained_weights(model_weights_name, cache_dir=cache_dir, offline=offline)

            version = self._load_version(model_type=model_weights_name, cache_dir=cache_dir)
        else:
            version = ""

        self.version = version

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
                Default value is None since we mostly don't have the target except for retrain.

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
