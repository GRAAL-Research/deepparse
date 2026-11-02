# pylint: disable=too-many-arguments, too-many-positional-arguments
import warnings
from typing import Dict, Tuple

import torch

from ..download_tools import MODEL2VEC_MODELS, MODEL_REPO_IDS, get_model2vec_model_type
from . import BPEmbSeq2SeqModel, FastTextSeq2SeqModel, Model2VecSeq2SeqModel, Seq2SeqModel
from .model_loader import ModelLoader


class ModelFactory:
    """
    A factory for creating neural network models that predict the tags from addresses.
    """

    def __init__(self, model_loader: ModelLoader) -> None:
        self.model_loader = model_loader

    def create(
        self,
        model_type: str,
        device: torch.device,
        output_size: int = 9,
        attention_mechanism: bool = False,
        path_to_retrained_model: str | None = None,
        pre_trained_weights: bool = True,
        offline: bool = False,
        verbose: bool = True,
        **seq2seq_kwargs: Dict,
    ) -> Tuple[Seq2SeqModel, str]:
        """
        Model creation method.

        Args:
            model_type (str): the type of the model to create. Valid options:
                - fasttext
                - bpemb
                - model2vec-8m, model2vec-32m and model2vec-multilingual
            device (~torch.device): The device to use for the prediction.
            output_size (int): The size of the prediction layers (i.e. the number of tags to predict). The default
                value is ``9``.
            attention_mechanism (bool): Either or not to use the attention mechanism. The default value is ``False``.
            path_to_retrained_model (Union[str, None]): The path to the retrained model to use for the seq2seq. The
                default value is ``None``.
            pre_trained_weights (bool): Whether to load pre-trained weights or return an untrained model.
                The `path_to_retrained_model` argument takes precedence if specified. The default value is ``True``.
            offline (bool): Whether or not the model is an offline or an online. The default value is ``False``.
            verbose (bool): Turn on/off the verbosity of the model. The default value is ``True``.

        Return:
            A tuple (``x``, ``y``) where ``x`` is a :class:`~Seq2SeqModel` and ``y`` is a string representing
            the model's version.
        """

        if "fasttext" in model_type or "fasttext-light" in model_type:
            model = FastTextSeq2SeqModel(
                output_size=output_size,
                attention_mechanism=attention_mechanism,
                **seq2seq_kwargs,
            )

        elif "bpemb" in model_type:
            model = BPEmbSeq2SeqModel(
                output_size=output_size,
                attention_mechanism=attention_mechanism,
                **seq2seq_kwargs,
            )

        elif "model2vec" in model_type:
            # The encoder input size is the Model2Vec embeddings dimension, unless a retrained model overrides it.
            seq2seq_kwargs.setdefault("input_size", MODEL2VEC_MODELS[get_model2vec_model_type(model_type)][1])
            model = Model2VecSeq2SeqModel(
                output_size=output_size,
                attention_mechanism=attention_mechanism,
                **seq2seq_kwargs,
            )

        else:
            raise NotImplementedError(f"""
                    There is no {model_type} network implemented. model_type should be either "fasttext", "bpemb"
                    or "model2vec".
            """)

        if path_to_retrained_model:
            model, version = self.model_loader.load_weights(model, path_to_retrained_model, device)

        elif pre_trained_weights and model_type in MODEL_REPO_IDS:
            model, version = self.model_loader.load_pre_trained_model(model, model_type, offline, verbose)

        elif pre_trained_weights:
            warnings.warn(
                f"There are no pre-trained weights for the {model_type} model yet. The model is randomly "
                "initialized and needs to be retrained before use.",
                category=UserWarning,
            )
            version = ""

        else:
            version = ""

        model.to_device(device)

        return model, version
