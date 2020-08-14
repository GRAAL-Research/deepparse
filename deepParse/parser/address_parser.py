import os
from json import load

from deepParse.converter.converter import TargetConverter
from deepParse.embeddings_model import FastTextEmbeddingsModel, BPEmbEmbeddingsModel
from deepParse.model import FastTextAddressSeq2SeqModel, BPEmbAddressSeq2SeqModel
from deepParse.tools import download_model
from deepParse.vectorizer import FastTextVectorizer, BPEmbVectorizer


class AddressTagger:
    """
    **For fastText, will download data in deepParse_data first time not seen in user root.
    """

    def __init__(self, model, device):
        pre_trained_tags = load(open("pre_trained_tags_to_idx.json", "r"))
        target_converter = TargetConverter(pre_trained_tags)

        if model == "fasttext" or model == "lightest":
            path = os.path.join(os.path.expanduser('~'), "deepParse_data")  # todo to validate if work
            os.makedirs(path, exist_ok=True)

            file_name = download_model("fr", saving_dir=path)
            embeddings_model = FastTextEmbeddingsModel(file_name)

            self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model, target_converter=target_converter)

            self.pre_trained_model = FastTextAddressSeq2SeqModel()

        elif model == "bpemb":
            embeddings_model = BPEmbEmbeddingsModel(lang="multi", vs=100000, dim=300)
            self.vectorizer = BPEmbVectorizer(embeddings_model=embeddings_model, target_converter=target_converter)

            self.pre_trained_model = BPEmbAddressSeq2SeqModel()
        elif model == "fasttext-att":
            pass
        elif model == "bpemb-att":
            pass
        elif model == "best":
            pass
