from . import BPEmbEmbeddingsModel, FastTextEmbeddingsModel, MagnitudeEmbeddingsModel, EmbeddingsModel
from .. import download_fasttext_embeddings, download_fasttext_magnitude_embeddings


class EmbeddingsModelFactory:
    """
    A factory for the creation of embeddings models.
    """

    def create(self, embedding_model_type: str, cache_dir: str, verbose: bool = True) -> EmbeddingsModel:
        """
        Embeddings model creation method.
        Args:
            embeddings_model_type (str): the type of the embeddings model to create. Valid options:
                - bpemb
                - fasttext
                - fasttext_magnitude
            cache_dir (str): Path to the cache directory where the embeddings model exists or is to be downloaded.
            verbose (bool): Wether or not to make the loading of the embeddings verbose.
        Return:
            An :class:`~EmbeddingsModel`
        """
        if "bpemb" in embedding_model_type:
            embeddings_model = BPEmbEmbeddingsModel(verbose=verbose, cache_dir=cache_dir)

        elif "fasttext" in embedding_model_type:
            if "fasttext-light" in embedding_model_type:
                file_name = download_fasttext_magnitude_embeddings(cache_dir=cache_dir, verbose=verbose)

                embeddings_model = MagnitudeEmbeddingsModel(file_name, verbose=verbose)
            else:
                file_name = download_fasttext_embeddings(cache_dir=cache_dir, verbose=verbose)

                embeddings_model = FastTextEmbeddingsModel(file_name, verbose=verbose)

        else:
            raise NotImplementedError(
                f"""
            The {embedding_model_type} embeddings model does not exist.
            Existing embeddings models are: bpemb, fasttext and fasttext_magnitude"""
            )

        return embeddings_model
