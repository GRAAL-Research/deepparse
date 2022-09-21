from ..embeddings_models import (
    BPEmbEmbeddingsModel,
    FastTextEmbeddingsModel,
    MagnitudeEmbeddingsModel,
    EmbeddingsModel,
)
from . import BPEmbVectorizer, FastTextVectorizer, MagnitudeVectorizer, Vectorizer


class VectorizerFactory:
    """
    A factory for the creation of vectorizers associated with specific embeddings models.
    """

    def create(self, embeddings_model: EmbeddingsModel) -> Vectorizer:
        """
        Vectorizer creation method.
        Args:
            embeddings_model_type (str): the type of the embeddings model to create. Valid options:
                - bpemb
                - fasttext
                - fasttext_magnitude
            cache_dir (str): Path to the cache directory where the embeddings model exists or is to be downloaded.
            verbose (bool): Wether or not to make the loading of the embeddings verbose.
        Return:
            A :class:`~Vectorizer`
        """
        if isinstance(embeddings_model, BPEmbEmbeddingsModel):
            vectorizer = BPEmbVectorizer(embeddings_model)

        elif isinstance(embeddings_model, FastTextEmbeddingsModel):
            vectorizer = FastTextVectorizer(embeddings_model)

        elif isinstance(embeddings_model, MagnitudeEmbeddingsModel):
            vectorizer = MagnitudeVectorizer(embeddings_model)

        else:
            raise NotImplementedError(
                f"""
            There's no vectorizer corresponding to the embeddings model type provided.
            Supported embedding models are: BPEmbEmbeddingsModel, FastTextEmbeddingsModel and MagnitudeEmbeddingsModel.
            """
            )

        return vectorizer
