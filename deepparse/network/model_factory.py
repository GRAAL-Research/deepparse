# pylint: disable=too-many-arguments
from typing import Dict, Union
import os
import warnings

import torch

from . import FastTextSeq2SeqModel, BPEmbSeq2SeqModel, Seq2SeqModel
from ..weights_tools import handle_weights_upload
from ..download_tools import download_weights, latest_version


class ModelFactory:
    """
    A factory for creating neural network models that predict the tags from addresses.
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
            device (~torch.device): The device to use for the prediction.
            output_size (int): The size of the prediction layers (i.e. the number of tags to predict). The default
                value is ``9``.
            attention_mechanism (bool): Either or not to use the attention mechanism. The default value is ``False``.
            path_to_retrained_model (Union[str, None]): The path to the retrained model to use for the seq2seq. The
                default value is ``None``.
            offline (bool): Whether or not the model is an offline or an online. The default value is ``False``.
            verbose (bool): Turn on/off the verbosity of the model. The default value is ``True``.

        Return:
            A :class:`~Seq2SeqModel`.
        """

        # TODO: clean up the loading logic
        ############
        pre_trained_weights = True ####
        model_weights_name = model_type

        if path_to_retrained_model is not None:
            all_layers_param = self._load_weights(path_to_retrained_model)

            version = "FineTunedModel" + self._load_version(model_type=model_weights_name, cache_dir=cache_dir)
        elif pre_trained_weights:
            # Means we use the pretrained weights
            all_layers_param = self._load_pre_trained_weights(model_weights_name, cache_dir=cache_dir, offline=offline, verbose=verbose)
            version = self._load_version(model_type=model_weights_name, cache_dir=cache_dir)
        else:
            version = ""
        ############

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

        else:
            raise NotImplementedError(
                f"""
                    There is no {model_type} network implemented. model_type should be either "fasttext" or "bpemb".
            """
            )


        ##############
        model.load_state_dict(all_layers_param)
        model.to_device(device)
        ##############

        return model

    def _load_pre_trained_weights(self, model_type: str, cache_dir: str, offline: bool, verbose: bool) -> None:
        """
        Method to download and resolve the loading (into the network) of the pre-trained weights.

        Args:
            model_type (str): The network pretrained weights to load.
            cache_dir (str): The path to the cached directory to use for downloading (and loading) the
                model weights.
            offline (bool): Whether the model is an offline or an online.
        """
        model_path = os.path.join(cache_dir, f"{model_type}.ckpt")

        if not offline:
            if not os.path.isfile(model_path):
                warnings.warn(
                    f"No pre-trained model where found in the cache directory {cache_dir}. Thus, we will"
                    "automatically download the pre-trained model.",
                    category=UserWarning,
                )
                download_weights(model_type, cache_dir, verbose=verbose)
            elif not latest_version(model_type, cache_path=cache_dir, verbose=verbose):
                if verbose:
                    warnings.warn(
                        "A new version of the pretrained model is available. The newest model will be downloaded.",
                        category=UserWarning,
                    )
                download_weights(model_type, cache_dir, verbose=verbose)

        return self._load_weights(path_to_model_torch_archive=model_path)

    def _load_weights(self, path_to_model_torch_archive: str) -> None:
        """
        Method to load (into the network) the weights.

        Args:
            path_to_model_torch_archive (str): The path to the fine-tuned model Torch archive.
        """
        all_layers_params = handle_weights_upload(
            path_to_model_to_upload=path_to_model_torch_archive#, device=self.device TODO: make sure to handle the device when refactoring this part
        )

        # All the time, our torch archive includes meta-data along with the model weights.
        all_layers_params = all_layers_params.get("address_tagger_model")
        return all_layers_params

    def _load_version(self, model_type: str, cache_dir: str) -> str:
        """
        Method to load the local hashed version of the model as an attribute.

        Args:
            model_type (str): The network pretrained weights to load.
            cache_dir (str): The path to the cached directory to use for downloading (and loading) the
                model weights.

        Return:
            The hash of the model.

        """
        with open(os.path.join(cache_dir, model_type + ".version"), encoding="utf-8") as local_model_hash_file:
            return local_model_hash_file.readline().strip()
