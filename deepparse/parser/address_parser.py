import os
import re
from typing import List, Union

import torch
from numpy.core.multiarray import ndarray

from .parsed_address import ParsedAddress
from .. import load_tuple_to_device
from ..converter import TagsConverter, data_padding
from ..converter.data_padding import bpemb_data_padding
from ..embeddings_models import BPEmbEmbeddingsModel
from ..embeddings_models import FastTextEmbeddingsModel
from ..fasttext_tools import download_fasttext_embeddings
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
        model (str): The network name to use, can be either:

            - fasttext (need ~9 GO of RAM to be used);
            - bpemb (need ~2 GO of RAM to be used);
            - lightest (less RAM usage) (equivalent to bpemb);
            - fastest (quicker to process one address) (equivalent to fasttext);
            - best (best accuracy performance) (equivalent to bpemb).

            The default value is 'best' for the most accurate model.
        device (Union[int, str, torch.device]): The device to use can be either:

            - a ``GPU`` index in int format (e.g. ``0``);
            - a complete device name in a string format (e.g. ``'cuda:0'``);
            - a :class:`~torch.torch.device` object;
            - ``'cpu'`` for a  ``CPU`` use.

            The default value is GPU with the index ``0`` if it exist, otherwise the value is ``CPU``.
        rounding (int): The rounding to use when asking the probability of the tags. The default value is 4 digits.

    Note:
        For both the networks, we will download the pre-trained weights and embeddings in the ``.cache`` directory
        for the root user. Also, one can download all the dependencies of our pre-trained model using the
        `deepparse.download` module (e.g. python -m deepparse.download fasttext) before sending it to a node without
        access to Internet.

    Note:
        Also note that the first time the fastText model is instantiated on a computer, we download the fastText
        pre-trained embeddings of 6.8 GO, and this process can be quite long (a couple of minutes).

    Note:
        The predictions tags are the following

            - "StreetNumber": for the street number
            - "StreetName": for the name of the street
            - "Unit": for the unit (such as apartment)
            - "Municipality": for the municipality
            - "Province": for the province or local region
            - "PostalCode": for the postal code
            - "Orientation": for the street orientation (e.g. west, east)
            - "GeneralDelivery": for other delivery information

    Example:

        .. code-block:: python

                address_parser = AddressParser(device=0) #on gpu device 0
                parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")

                address_parser = AddressParser(model="fasttext", device="cpu") # fasttext model on cpu
                parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
    """

    def __init__(self, model: str = "best", device: Union[int, str, torch.device] = 0, rounding: int = 4) -> None:
        self._process_device(device)

        self.rounding = rounding

        self.tags_converter = TagsConverter(_pre_trained_tags_to_idx)

        model = model.lower()
        if model in ("fasttext", "fastest"):
            path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")
            os.makedirs(path, exist_ok=True)

            file_name = download_fasttext_embeddings("fr", saving_dir=path)
            embeddings_model = FastTextEmbeddingsModel(file_name)

            self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model)

            self.data_converter = data_padding

            self.pre_trained_model = PreTrainedFastTextSeq2SeqModel(self.device)

        elif model in ("bpemb", "best", "lightest"):
            self.vectorizer = BPEmbVectorizer(embeddings_model=BPEmbEmbeddingsModel(lang="multi", vs=100000, dim=300))

            self.data_converter = bpemb_data_padding

            self.pre_trained_model = PreTrainedBPEmbSeq2SeqModel(self.device)
        else:
            raise NotImplementedError(f"There is no {model} network implemented. Value can be: "
                                      f"fasttext, bpemb, lightest (bpemb), fastest (fasttext) or best (bpemb).")

        self.pre_trained_model.eval()

    def __call__(self,
                 addresses_to_parse: Union[List[str], str],
                 with_prob: bool = False) -> Union[ParsedAddress, List[ParsedAddress]]:
        """
        Callable method to parse the components of an address or a list of address.

        Args:
            addresses_to_parse (Union[list[str], str]): The addresses to be parse, can be either a single address
                (when using str) or a list of address. When using a list of addresses, the addresses are processed in
                batch, allowing a faster process. For example, using fastText model, a single address takes around
                0.003 seconds to be parsed using a batch of 1 (1 element at the time is processed).
                This time can be reduced to 0.00035 seconds per address when using a batch of 128
                (128 elements at the time are processed).
            with_prob (bool): If true, return the probability of all the tags with the specified
                rounding.

        Return:
            Either a :class:`~deepparse.deepparse.parsed_address.ParsedAddress` or a list of
            :class:`~deepparse.deepparse.parsed_address.ParsedAddress` when given more than one address.


        Example:

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
                    parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6", with_prob=True)

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
                                          addresses_to_parse: List[str],
                                          with_prob: bool) -> Union[ParsedAddress, List[ParsedAddress]]:
        """
        Method to fill the mapping for every address between a address components and is associated predicted tag (or
        tag and prob).
        """
        tagged_addresses_components = []

        for address_to_parse, tags_prediction, tags_prediction_prob in zip(addresses_to_parse, tags_predictions,
                                                                           tags_predictions_prob):
            tagged_address_components = []
            for word, predicted_idx_tag, tag_proba in zip(address_to_parse.split(), tags_prediction,
                                                          tags_prediction_prob):
                tag = self.tags_converter(predicted_idx_tag)
                if with_prob:
                    tag = (tag, round(tag_proba, self.rounding))
                tagged_address_components.append((word, tag))
            tagged_addresses_components.append(ParsedAddress({address_to_parse: tagged_address_components}))

        if len(tagged_addresses_components) == 1:
            return tagged_addresses_components[0]
        return tagged_addresses_components

    def _process_device(self, device: Union[int, str, torch.device]):
        """
        Function to process the device depending of the argument type.

        Set the device as a torch device object.
        """
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            if re.fullmatch(r"cpu|cuda:\d+", device.lower()):
                self.device = torch.device(device)
            else:
                raise ValueError("String value should be 'cpu' or follow the pattern 'cuda:[int]'.")
        elif isinstance(device, int):
            if device >= 0:
                self.device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")
            else:
                raise ValueError("Device should not be a negative number.")
        else:
            raise ValueError("Device should be a string, an int or a torch device.")
