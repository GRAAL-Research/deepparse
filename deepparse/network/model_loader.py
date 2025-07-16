from typing import Tuple

import torch

from . import Seq2SeqModel
from ..download_tools import download_weights, load_version
from ..weights_tools import handle_weights_upload


class ModelLoader():
    def __init__(self, cache_dir: str) -> None:
        """
        Class to download and/or load model weights.
        Args:
            cache_dir (str): The path to the cached directory to use for downloading (and loading) the
                model weights.
        """
        self.cache_dir = cache_dir

    def load_pre_trained_model(self, model: Seq2SeqModel, model_type: str, offline: bool, verbose: bool) -> Tuple[Seq2SeqModel, str]:
        """
        Method to download and resolve the loading (into the network) of the pre-trained weights.

        Args:
            model (Seq2SeqModel): The model to be loaded with pre-trained weights.
            model_type (str): The network pretrained weights to load.
            offline (bool): Whether the model is an offline or an online.
            verbose (bool): Turn on/off the verbosity of the model. The default value is ``True``.
        Return:
            A tuple (``x``, ``y``) where ``x`` is a :class:`~Seq2SeqModel` and ``y`` is a string representing the model's version.

        """

        model_id = download_weights(model_type, saving_dir=self.cache_dir, verbose=verbose, offline=offline)

        model = model.from_pretrained(model_id, local_files_only=True, cache_dir=self.cache_dir)

        version = load_version(model_type, self.cache_dir)

        return model, version

    def load_weights(self, model: Seq2SeqModel, path_to_model_torch_archive: str, device: torch.device) -> Tuple[Seq2SeqModel, str]:
        """
        Method to load (into the network) the weights.

        Args:
            model (Seq2SeqModel): The model to be loaded with pre-trained weights.
            path_to_model_torch_archive (str): The path to the fine-tuned model Torch archive.
            device (~torch.device): The device to use for the prediction.

        Return:
            A tuple (``x``, ``y``) where ``x`` is a :class:`~Seq2SeqModel` and ``y`` is a string representing the model's version.

        """
        all_layers_params = handle_weights_upload(
            path_to_model_to_upload=path_to_model_torch_archive, device=device
        )

        # All the time, our torch archive includes meta-data along with the model weights.
        all_layers_params = all_layers_params.get("address_tagger_model")

        model.load_state_dict(all_layers_params)

        version = all_layers_params["version"]

        return model, version

