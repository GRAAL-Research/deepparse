from . import WordEmbeddingPadder, SubwordEmbeddingPadder


class DataPadderFactory:
    def create(self, embedding_model_type, padding_value):
        if embedding_model_type == "fasttext" or embedding_model_type == "fasttext_magnitude":
            padder = WordEmbeddingPadder(padding_value)

        elif embedding_model_type == "bpemb":
            padder = SubwordEmbeddingPadder(padding_value)

        else:
            raise NotImplementedError(
                f"""
            There's no padder corresponding to the {embedding_model_type} embedding model type.
            Supported embedding models are: bpemb, fasttext and fasttext_magnitude.
            """
            )

        return padder
