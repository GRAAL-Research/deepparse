from ..embeddings_models import BPEmbEmbeddingsModel, FastTextEmbeddingsModel, MagnitudeEmbeddingsModel
from . import BPEmbVectorizer, FastTextVectorizer, MagnitudeVectorizer


class VectorizerFactory:
    def __init__(self, embeddings_model_factory) -> None:
        self.embeddings_model_factory = embeddings_model_factory

    def create(self, embedding_model_type, cache_dir, verbose=True):
        embeddings_model = self.embeddings_model_factory.create(embedding_model_type, cache_dir, verbose)

        vectorizer = None
        if isinstance(embeddings_model, BPEmbEmbeddingsModel):
            vectorizer = BPEmbVectorizer(embeddings_model)

        elif isinstance(embeddings_model, FastTextEmbeddingsModel):
            vectorizer = FastTextVectorizer(embeddings_model)

        elif isinstance(embeddings_model, MagnitudeEmbeddingsModel):
            vectorizer = MagnitudeVectorizer(embeddings_model)

        else:
            raise NotImplementedError(f"""
            There's no vectorizer corresponding to the {embedding_model_type} embedding model type.
            Supported embedding models are: bpemb, fasttext and fasttext_magnitude.
            """)

        return vectorizer
