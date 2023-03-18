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
            embeddings_model (:class:`~EmbeddingsModel`): The embeddings model for which a vectorizer is to be created
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
                """
            There's no vectorizer corresponding to the embeddings model type provided.
            Supported embedding models are: BPEmbEmbeddingsModel, FastTextEmbeddingsModel and MagnitudeEmbeddingsModel.
            """
            )

        return vectorizer
