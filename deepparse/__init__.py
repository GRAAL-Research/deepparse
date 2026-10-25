# pylint: disable=wildcard-import
# isort: skip_file
from .bpemb_loader import *
from .bpemb_url_bug_fix import *
from .download_tools import *
from .validations import *
from .version import __version__
from .weights_tools import *

# Legacy attributes: deepparse helpers and constants that the former wildcard imports (without __all__) re-exported.
# They are not part of the public API (they are not in __all__) but stay reachable as deepparse.<name> until 1.0.0 so
# that no existing code breaks. The standard library and third-party modules re-exported by accident (os, torch, np,
# typing aliases, etc.) are no longer reachable from deepparse.
from .bpemb_loader import load_word2vec_binary_vectors
from .download_tools import (
    FASTTEXT_COPYRIGHT_MIT_LICENSE,
    HTTP_CLIENT_ERROR_STATUS_CODE,
    MODEL_REPO_IDS,
    NEXT_RANGE_STATUS_CODE,
    PYARROW_AVAILABLE,
    WORLDWIDE_ADDRESSES_REMOVED_TAGS,
    WORLDWIDE_ADDRESSES_REPO_ID,
    WORLDWIDE_ADDRESSES_TAGS_MAPPING,
    download_gz_model,
)
from .validations import (
    DataError,
    extract_package_version,
    valid_poutyne_version,
    validate_data_to_parse,
    validate_if_any_empty,
    validate_if_any_none,
    validate_if_any_whitespace_only,
)
from .weights_tools import handle_weights_upload, weights_init

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
    # Deprecated, will be removed in 1.0.0.
    "BPEmbBaseURLWrapperBugFix",
    "FASTTEXT_AVAILABLE",
    "load_fasttext_embeddings",
]
