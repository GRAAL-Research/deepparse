from . import BPEmbEmbeddingsModel, FastTextEmbeddingsModel, MagnitudeEmbeddingsModel
from .. import download_fasttext_embeddings, download_fasttext_magnitude_embeddings


class EmbeddingsModelFactory:
    def create(self, embedding_model_type, cache_dir, verbose=True):
        if embedding_model_type == "bpemb":
            embeddings_model = BPEmbEmbeddingsModel(verbose=verbose, cache_dir=cache_dir)

        elif embedding_model_type == "fasttext":
            file_name = download_fasttext_embeddings(cache_dir=cache_dir, verbose=verbose)

            embeddings_model = FastTextEmbeddingsModel(file_name, verbose=verbose)

        elif embedding_model_type == "fasttext_magnitude":
            file_name = download_fasttext_magnitude_embeddings(cache_dir=cache_dir, verbose=verbose)

            embeddings_model = MagnitudeEmbeddingsModel(file_name, verbose=verbose)

        else:
            raise NotImplementedError(
                f"""
            The {embedding_model_type} embeddings model does not exist.
            Existing embeddings models are: bpemb, fasttext and fasttext_magnitude"""
            )

        return embeddings_model
