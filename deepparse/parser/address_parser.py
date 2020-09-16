import os
from typing import List, Union, Dict

import torch
from numpy.core.multiarray import ndarray

from .. import load_tuple_to_device
from ..converter import TagsConverter, data_padding
from ..converter.data_padding import bpemb_data_padding
from ..embeddings_models import FastTextEmbeddingsModel
from ..embeddings_models.bp_embeddings_model import BPEmbEmbeddingsModel
from ..fasttext_tools import download_fasttext_model
from ..network.pre_trained_bpemb_seq2seq import PreTrainedBPEmbSeq2SeqModel
from ..network.pre_trained_fasttext_seq2seq import PreTrainedFastTextSeq2SeqModel
from ..vectorizer import FastTextVectorizer, BPEmbVectorizer

_pre_trained_tags_to_idx = {
    "StreetNumber": 0,
    "StreetName": 1,
    "Unit": 2,
    "Municipality": 3,
    "Province": 4,
    "PostalCode": 5,
    "Orientation": 6,
    "GeneralDelivery": 7,
    "EOS": 8  # the 9th is the EOS with idx 8
}


class AddressParser:
    """
    Address parser to parse an address or a list of address using one of the seq2seq pre-trained
    networks either with fastText or BPEmb.

    Args:
        model (str): The network name to use, can be either fasttext, bpemb, lightest (equivalent to fasttext) or
            best (equivalent to bpemb).
        device (Union[int, str]): The device to use can be either a ``GPU`` index (e.g. 0) in int format or string
            format or ``'CPU'``.
        rounding (int): The rounding to use when asking the probability of the tags. The default value is 4 digits.

    Note:
        For both the networks, we will download the pre-trained weights and embeddings in the ``.cache`` directory
        for the root user.
    """

    def __init__(self, model: str, device: Union[int, str], rounding: int = 4) -> None:
        if device in "cpu":
            self.device = device
        else:
            self.device = "cuda:%d" % int(device)
        self.rounding = rounding

        self.tags_converter = TagsConverter(_pre_trained_tags_to_idx)

        if model in "fasttext" or model in "lightest":
            path = os.path.join(os.path.expanduser('~'), ".cache/deepparse")
            os.makedirs(path, exist_ok=True)

            file_name = download_fasttext_model("fr", saving_dir=path)
            embeddings_model = FastTextEmbeddingsModel(file_name)

            self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model)

            self.data_converter = data_padding

            self.pre_trained_model = PreTrainedFastTextSeq2SeqModel(self.device)

        elif model in "bpemb" or model in "best":
            self.vectorizer = BPEmbVectorizer(embeddings_model=BPEmbEmbeddingsModel(lang="multi", vs=100000, dim=300))

            self.data_converter = bpemb_data_padding

            self.pre_trained_model = PreTrainedBPEmbSeq2SeqModel(self.device)
        else:
            raise NotImplementedError(f"There is no {model} network implemented. Value can be: "
                                      f"fasttext, bpemb, lightest (fastext) or best (bpemb).")

    def __call__(self, addresses_to_parse: Union[List[str], str], with_prob: bool = False) -> Dict:
        """
        Callable method to parse the components of an address or a list of address.

        Args:
            addresses_to_parse (Union[list[str], str]): The addresses to be parse, can be either a single address
                (when using str) or a list of address.
            with_prob (bool): If true, return the probability of all the tags with the specified
                rounding.

        Return:
            A dictionary where the keys are the parsed address and the values dictionary. For the second
            dictionary: the key are the address components (e.g. a street number such as 305) and the value are
            either the tag of the components (e.g. StreetName) or a tuple (``x``, ``y``) where ``x`` is the tag and
            ``y`` is the probability (e.g. 0.9981).

        """
        if isinstance(addresses_to_parse, str):
            addresses_to_parse = [addresses_to_parse]

        # since training data is lowercase
        lower_cased_addresses_to_parse = [address.lower() for address in addresses_to_parse]

        vectorize_address = self.vectorizer(lower_cased_addresses_to_parse)

        padded_address = self.data_converter(vectorize_address)
        padded_address = load_tuple_to_device(padded_address, self.device)

        predictions = self.pre_trained_model(*padded_address)

        tags_predictions = predictions.max(2)[1].transpose(0, 1).cpu().numpy()
        tags_predictions_prob = torch.exp(predictions.max(2)[0]).transpose(0, 1).detach().cpu().numpy()

        tagged_addresses_components = self._fill_tagged_addresses_components(tags_predictions, tags_predictions_prob,
                                                                             addresses_to_parse, with_prob)

        return tagged_addresses_components

    def _fill_tagged_addresses_components(self, tags_predictions: ndarray, tags_predictions_prob: ndarray,
                                          addresses_to_parse: List[str], with_prob: bool) -> Dict:
        """
        Method to fill the mapping for every address between a address components and is associated predicted tag (or
        tag and prob).
        """
        tagged_addresses_components = {}

        for idx, (address_to_parse, tags_prediction,
                  tags_prediction_prob) in enumerate(zip(addresses_to_parse, tags_predictions, tags_predictions_prob)):
            tagged_address_components = {}
            for word, predicted_idx_tag, tag_proba in zip(address_to_parse.split(), tags_prediction,
                                                          tags_prediction_prob):
                tag = self.tags_converter(predicted_idx_tag)
                if with_prob:
                    tag = (tag, round(tag_proba, self.rounding))
                tagged_address_components[word] = tag
            tagged_addresses_components[addresses_to_parse[idx]] = tagged_address_components

        return tagged_addresses_components
