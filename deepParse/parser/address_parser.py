import os
from json import load
from typing import List, Union

from deepParse.converter import TagsConverter, data_padding
from deepParse.converter.data_padding import bpemb_data_padding
from deepParse.embeddings_model import FastTextEmbeddingsModel
from deepParse.embeddings_model.bp_embeddings_model import BPEmbEmbeddingsModel
from deepParse.model import PretrainedFastTextSeq2SeqModel, PretrainedBPEmbSeq2SeqModel
from deepParse.tools import download_fasttext_model
from deepParse.vectorizer import FastTextVectorizer, BPEmbVectorizer


class AddressParser:
    """
    **For fastText, will download data in deepParse_data first time not seen in user root.
    """

    def __init__(self, model: str, device: Union[int, str]):
        self.device = "cuda:%d" % str(device)

        self.tags_converter = TagsConverter(load(open("pre_trained_tags_to_idx.json", "r")))

        if model == "fasttext" or model == "lightest":
            path = os.path.join(os.path.expanduser('~'), "deepParse_data")
            os.makedirs(path, exist_ok=True)

            file_name = download_fasttext_model("fr", saving_dir=path)
            embeddings_model = FastTextEmbeddingsModel(file_name)

            self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model)

            self.data_converter = data_padding

            self.pre_trained_model = PretrainedFastTextSeq2SeqModel(device)

        elif model == "bpemb":
            self.vectorizer = BPEmbVectorizer(embeddings_model=BPEmbEmbeddingsModel(lang="multi", vs=100000, dim=300))

            self.data_converter = bpemb_data_padding

            self.pre_trained_model = PretrainedBPEmbSeq2SeqModel(device)

        elif model == "fasttext-att":
            pass
        elif model == "bpemb-att":
            pass
        elif model == "best":
            pass
        elif model == "lightest":
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
