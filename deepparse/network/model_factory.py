# pylint: disable=too-many-arguments
from typing import Dict, Union

import torch

from . import FastTextSeq2SeqModel, BPEmbSeq2SeqModel, Seq2SeqModel


class ModelFactory:
    """
    A factory for the creation of neural network models that predict the tags from addresses
    """

    def create(
        self,
        model_type: str,
        cache_dir: str,
        device: torch.device,
        output_size: int = 9,
        attention_mechanism: bool = False,
        path_to_retrained_model: Union[str, None] = None,
        offline: bool = False,
        verbose: bool = True,
        **seq2seq_kwargs: Dict,
    ) -> Seq2SeqModel:
        """
        Model creation method.

        Args:
            model_type (str): the type of the model to create. Valid options:
                - fasttext
                - bpemb
            cache_dir (str): The path to the cached directory to use for downloading (and loading) the
                model weights.
            device (~torch.device): The device tu use for the prediction.
            output_size (int): The size of the prediction layers (i.e. the number of tag to predict).
            attention_mechanism (bool): Either or not to use attention mechanism. The default value is False.
            path_to_retrained_model (Union[str, None]): The path to the retrained model to use for the seq2seq.
            offline (bool): Wether or not the model is an offline or an online.
            verbose (bool): Turn on/off the verbosity of the model. The default value is True.

        Return:
            A :class:`~Seq2SeqModel`.
        """
        if "fasttext" in model_type or "fasttext-light" in model_type:
            model = FastTextSeq2SeqModel(
                cache_dir=cache_dir,
                device=device,
                output_size=output_size,
                verbose=verbose,
                path_to_retrained_model=path_to_retrained_model,
                attention_mechanism=attention_mechanism,
                offline=offline,
                **seq2seq_kwargs,
            )

        elif "bpemb" in model_type:
            model = BPEmbSeq2SeqModel(
                cache_dir=cache_dir,
                device=device,
                output_size=output_size,
                verbose=verbose,
                path_to_retrained_model=path_to_retrained_model,
                attention_mechanism=attention_mechanism,
                offline=offline,
                **seq2seq_kwargs,
            )

        else:
            raise NotImplementedError(
                f"""
                    There is no {model_type} network implemented. model_type should be either fasttext or bpemb
            """
            )

        return model
