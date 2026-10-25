# pylint: disable=wildcard-import
# isort: skip_file
from .bpemb_loader import *
from .bpemb_url_bug_fix import *
from .download_tools import *
from .validations import *
from .version import __version__
from .weights_tools import *

# Legacy attributes: names that the former wildcard imports (without __all__) re-exported by accident, such as
# standard library and third-party modules, typing aliases and imported helpers. They are not part of the public API
# (they are not in __all__) but stay reachable as deepparse.<name> so that no existing code breaks.
from .bpemb_loader import np, re, requests, sentencepiece, shutil, tarfile, tempfile
from .download_tools import (
    Dict,
    HfFileSystem,
    List,
    Path,
    Tuple,
    cached_file,
    disable_progress_bar,
    enable_progress_bar,
    extract_commit_hash,
    gzip,
    hf_hub_download,
    importlib,
    logger,
    logging,
    os,
    pq,
    snapshot_download,
    sys,
    urlopen,
    warnings,
)
from .validations import (
    DataError,
    poutyne,
    validate_if_any_empty,
    validate_if_any_none,
    validate_if_any_whitespace_only,
)
from .weights_tools import CloudPath, OrderedDict, S3Path, Union, init, nn, torch

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
    "download_models",
    "download_weights",
    "download_worldwide_addresses_sample",
    "load_version",
    # Deprecated, will be removed in 1.0.0.
    "BPEmbBaseURLWrapperBugFix",
    "FASTTEXT_AVAILABLE",
    "load_fasttext_embeddings",
]
