# isort: skip_file
from .bpemb_loader import BPEmbLoader
from .download_tools import (
    CACHE_PATH,
    MODEL_MAPPING_CHOICES,
    download_fasttext_embeddings,
    download_fasttext_magnitude_embeddings,
    download_model,
    download_model2vec_embeddings,
    download_models,
    download_weights,
    download_worldwide_addresses_sample,
    load_version,
)
from .version import __version__

# The submodules stay reachable as deepparse.<name> (e.g. deepparse.weights_tools), but their helpers and constants are
# not part of the public API and must be imported from their own module.
from . import validations, weights_tools

# The public API of the deepparse top-level package. The parsers, dataset containers and other components are in
# their subpackages (e.g. deepparse.parser.AddressParser).
__all__ = [
    # BPEmb embeddings loader.
    "BPEmbLoader",
    # Pretrained models and embeddings download.
    "CACHE_PATH",
    "MODEL_MAPPING_CHOICES",
    "download_fasttext_embeddings",
    "download_fasttext_magnitude_embeddings",
    "download_model",
    "download_model2vec_embeddings",
    "download_models",
    "download_weights",
    "download_worldwide_addresses_sample",
    "load_version",
]
