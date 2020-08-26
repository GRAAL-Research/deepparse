import os
from typing import List, Union

from ..converter import TagsConverter, data_padding
from ..converter.data_padding import bpemb_data_padding
from ..embeddings_model import FastTextEmbeddingsModel
from ..embeddings_model.bp_embeddings_model import BPEmbEmbeddingsModel
from ..fasttest_tools import download_fasttext_model
from ..model import PretrainedFastTextSeq2SeqModel, PretrainedBPEmbSeq2SeqModel
from ..vectorizer import FastTextVectorizer, BPEmbVectorizer

_pre_trained_tags_to_idx = {
    "StreetNumber": 0,
    "StreetName": 1,
    "Unit": 2,
    "Municipality": 3,
    "Province": 4,
    "PostalCode": 5,
    "Orientation": 6,
    "GeneralDelivery": 7
}  # the 9th is the EOS with idx 8


class AddressParser:
    """
    **For fastText, will download data in deepParse_data first time not seen in user root.
    """

    def __init__(self, model: str, device: Union[int, str]):
        self.device = "cuda:%d" % int(device)

        self.tags_converter = TagsConverter(_pre_trained_tags_to_idx)

        if model == "fasttext" or model == "lightest":
            path = os.path.join(os.path.expanduser('~'), ".cache/deepParse")
            os.makedirs(path, exist_ok=True)

            file_name = download_fasttext_model("fr", saving_dir=path)
            embeddings_model = FastTextEmbeddingsModel(file_name)

            self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model)

            self.data_converter = data_padding

            self.pre_trained_model = PretrainedFastTextSeq2SeqModel(self.device)

        elif model == "bpemb":
            self.vectorizer = BPEmbVectorizer(embeddings_model=BPEmbEmbeddingsModel(lang="multi", vs=100000, dim=300))

            self.data_converter = bpemb_data_padding

            self.pre_trained_model = PretrainedBPEmbSeq2SeqModel(self.device)

        elif model == "fasttext-att":
            pass
        elif model == "bpemb-att":
            pass
        elif model == "best":
            pass

    def __call__(self, address_to_parse: Union[List[str], str]):
        if isinstance(str, address_to_parse):
            address_to_parse = list(address_to_parse)

        # todo send to device
        self.vectorizer(address_to_parse)  # input must be a list

        # todo step pour convertir pour la "forward pass"
        self.data_converter()

        prediction = self.pre_trained_model(...)

        # get max prob
        # convert to tag
