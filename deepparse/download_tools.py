import gzip
import importlib.util
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

try:
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    pq = None
    PYARROW_AVAILABLE = False

from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from transformers.utils.hub import cached_file, extract_commit_hash
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from .bpemb_url_bug_fix import BPEmbBaseURLWrapperBugFix

# Kept for backward compatibility: Deepparse no longer needs the fasttext package to load the FastText embeddings.
FASTTEXT_AVAILABLE = importlib.util.find_spec("fasttext") is not None

logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")

# Status code starting in the 4xx are client error status code.
# That is Deepparse, server problem (e.g. Deepparse server is offline).
HTTP_CLIENT_ERROR_STATUS_CODE = 400
# Status code starting in the 5xx are the next range status code.
NEXT_RANGE_STATUS_CODE = 500

MODEL_MAPPING_CHOICES: Dict[str, str] = {
    "fasttext": "fasttext",
    "fasttext-attention": "fasttext_attention",
    "fasttext-light": "fasttext",
    "bpemb": "bpemb",
    "bpemb-attention": "bpemb_attention",
    "model2vec-8m": "model2vec-8m",
    "model2vec-8m-attention": "model2vec-8m_attention",
    "model2vec-32m": "model2vec-32m",
    "model2vec-32m-attention": "model2vec-32m_attention",
    "model2vec-multilingual": "model2vec-multilingual",
    "model2vec-multilingual-attention": "model2vec-multilingual_attention",
}

MODEL_REPO_IDS = {
    "bpemb": "deepparse/bpemb-base",
    "bpemb_attention": "deepparse/bpemb-attention",
    "fasttext": "deepparse/fasttext-base",
    "fasttext_attention": "deepparse/fasttext-attention",
    "fasttext-light": "deepparse/fasttext-base",
    "fasttext-light_attention": "deepparse/fasttext-attention",
}

WORLDWIDE_ADDRESSES_REPO_ID = "deepparse/worldwide-addresses"

# The worldwide-addresses dataset tags that are not deepparse tags. The words tagged with a removed tag are removed
# from the address.
WORLDWIDE_ADDRESSES_TAGS_MAPPING: Dict[str, str] = {"Suburb": "Municipality", "District": "Municipality"}
WORLDWIDE_ADDRESSES_REMOVED_TAGS = ("County", "Country")

# Model2Vec model types with their Hugging Face embeddings repository and embeddings dimension.
MODEL2VEC_MODELS: Dict[str, Tuple[str, int]] = {
    "model2vec-8m": ("minishlab/potion-base-8M", 256),
    "model2vec-32m": ("minishlab/potion-base-32M", 512),
    "model2vec-multilingual": ("minishlab/potion-multilingual-128M", 256),
}


def get_model2vec_model_type(model_type: str) -> str:
    """
    Function to get the Model2Vec base model type (e.g. ``model2vec-8m``) of a model type that can include the
    attention suffix (e.g. ``model2vec-8m_attention``).
    """
    for model2vec_model_type in MODEL2VEC_MODELS:
        if model_type.startswith(model2vec_model_type):
            return model2vec_model_type
    raise ValueError(f"{model_type} is not a Model2Vec model type. Valid ones are {list(MODEL2VEC_MODELS)}.")


def download_model2vec_embeddings(
    model_type: str, cache_dir: str, verbose: bool = True, offline: bool = False
) -> Tuple[str, int]:
    """
    Function to download the Model2Vec embeddings of a model2vec model type.

    Return the local path to the Model2Vec embeddings and the embeddings dimension.
    """
    repo_id, dim = MODEL2VEC_MODELS[get_model2vec_model_type(model_type)]

    if verbose and not offline:
        logger.info("Downloading the Model2Vec embeddings %s.", repo_id)

    # We only download the files Model2Vec loads to skip the ONNX export, which doubles the download size.
    local_embeddings_path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        local_files_only=offline,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json"],
    )

    return local_embeddings_path, dim


def download_fasttext_magnitude_embeddings(cache_dir: str, verbose: bool = True, offline: bool = False) -> str:
    """
    Function to download the magnitude pretrained FastText model.

    Return the full path to the FastText embeddings.
    """

    try:
        local_embeddings_file_path = cached_file(
            "deepparse/fasttext-base",
            filename="fasttext.magnitude",
            revision="light-embeddings",
            local_files_only=True,
            cache_dir=cache_dir,
        )
    except OSError:
        if verbose:
            logger.info(
                "The FastText pretrained word embeddings will be downloaded in magnitude format (3.5 GO), "
                "this process will take several minutes."
            )

        local_embeddings_file_path = hf_hub_download(
            "deepparse/fasttext-base",
            filename="fasttext.magnitude",
            revision="light-embeddings",
            cache_dir=cache_dir,
            local_files_only=offline,
        )

    return local_embeddings_file_path


def download_weights(model_type: str, saving_dir: str, verbose: bool = True, offline: bool = False) -> str:
    """
    Function to download the pretrained weights of one of our pre-trained base models.
    Args:
        model_type (str): The network pretrained weights to load.
        saving_dir: The path to the saving directory.
        verbose (bool): Either or not to be verbose during the download of a model. The default value is ``True``.
        offline (bool): Whether the model is an offline or an online.
    Return:
        The model rep id (str) which can be used with hugging face's `from_pretrained` method.
    """
    repo_id = MODEL_REPO_IDS[model_type]

    if not offline:
        if verbose:
            warnings.warn(
                "The offline parameter is set to False, so if a "
                f"new pre-trained `{model_type}` model is available it will "
                "automatically be downloaded.",
                category=UserWarning,
            )

        # Disabling progress bar since it shows up even when no files are up to date which can get confusing
        disable_progress_bar()

        snapshot_download(repo_id, cache_dir=saving_dir, local_files_only=offline)

        # Re-enabling progress bar
        enable_progress_bar()

    return repo_id


def load_version(model_type: str, cache_dir: str) -> str:
    """
    Method to load the local hashed version of the model as an attribute.

    Args:
        model_type (str): The network pretrained weights to load.
        cache_dir (str): The path to the cached directory to use for downloading (and loading) the
            model weights.

    Return:
        The hash of the model which corresponds to the hash of the latest commit in the local revision.
    """
    repo_id = MODEL_REPO_IDS[model_type]

    config_file = cached_file(repo_id, "config.json", local_files_only=True, cache_dir=cache_dir)

    version = extract_commit_hash(config_file, None)

    return version


def download_models(saving_cache_path: Path | None = None) -> None:
    """
    Function to download all the pretrained models.  It will download all the model's checkpoints and version files.

    Args:
        saving_cache_path: The path to the saving cache directory for the specified model.
    """
    for model_type in MODEL_MAPPING_CHOICES:
        download_model(model_type, saving_cache_path=saving_cache_path)


def download_model(
    model_type: str,
    saving_cache_path: Path | None = None,
) -> None:
    """
    Function to download a pretrained model. It will download its corresponding checkpoint and version file.

    Args:
        model_type: The model type (i.e. ``fasttext`` or ``bpemb-attention``).
        saving_cache_path: The path to the saving cache directory for the specified model.
    """

    if saving_cache_path is None:
        # We use the default cache path '~/.cache/deepparse'.
        saving_cache_path = CACHE_PATH

    if "fasttext" in model_type and "fasttext-light" not in model_type:
        download_fasttext_embeddings(cache_dir=saving_cache_path)
    elif model_type == "fasttext-light":
        download_fasttext_magnitude_embeddings(cache_dir=saving_cache_path)
    elif "bpemb" in model_type:
        BPEmbBaseURLWrapperBugFix(
            lang="multi", vs=100000, dim=300, cache_dir=saving_cache_path
        )  # The class manages the download of the pretrained words embedding
    elif "model2vec" in model_type:
        download_model2vec_embeddings(model_type, cache_dir=saving_cache_path)

    model_type = MODEL_MAPPING_CHOICES[model_type]

    download_weights(model_type, saving_cache_path, verbose=True, offline=False)


def download_worldwide_addresses_sample(country_code: str = "ca", num_samples: int = 1000) -> List[Tuple[str, List]]:
    """
    Function to download a small sample of tagged addresses of one country from the `worldwide-addresses
    <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_ HuggingFace dataset and convert it to the
    deepparse tag space. It is meant to give small, ready-to-use datasets for examples and quick experiments.

    Only the needed part of the country's last parquet file is read (the beginning of each country's data mostly
    holds postal-code-only addresses, while the end holds complete street addresses). The tags are mapped to the
    deepparse tags as follows: ``Suburb`` and ``District`` become ``Municipality``, and the words tagged ``County``
    or ``Country`` are removed from the address. The addresses left empty by this removal are dropped.

    It requires the ``pyarrow`` package (``pip install pyarrow`` or ``pip install deepparse[all]``).

    Args:
        country_code (str): The `ISO 3166-1 alpha-2 <https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2>`_ code of the
            country to download (e.g. ``"ca"`` or ``"us"``). The default value is ``"ca"``.
        num_samples (int): The maximum number of tagged addresses to return. The default value is ``1000``.

    Return:
        A list of tuples ``(address, tags)`` where the address is a string and the tags a list of deepparse tags, one
        per whitespace-separated word. The list can be used as is with a
        :class:`~deepparse.dataset_container.ListDatasetContainer`.
    """
    if not PYARROW_AVAILABLE:
        raise ImportError(
            "The 'pyarrow' package is needed to read the worldwide-addresses dataset. Install it with "
            "'pip install pyarrow' or 'pip install deepparse[all]'."
        )
    if num_samples <= 0:
        raise ValueError(f"The number of samples must be a positive integer, got {num_samples}.")

    country_code = country_code.lower()
    file_system = HfFileSystem()
    chunk_paths = file_system.glob(f"datasets/{WORLDWIDE_ADDRESSES_REPO_ID}/{country_code}/chunk-*.parquet")
    if not chunk_paths:
        raise ValueError(
            f"No data found for the country code '{country_code}' in the {WORLDWIDE_ADDRESSES_REPO_ID} dataset. "
            "Use an ISO 3166-1 alpha-2 country code (e.g. 'ca')."
        )
    last_chunk_path = max(chunk_paths, key=lambda path: int(path.rsplit("chunk-", 1)[1].split(".", 1)[0]))

    samples: List[Tuple[str, List]] = []
    with file_system.open(last_chunk_path, "rb") as parquet_file:
        parquet_reader = pq.ParquetFile(parquet_file)
        # We read the row groups from the end of the file until we have enough samples.
        for row_group_index in reversed(range(parquet_reader.num_row_groups)):
            row_group = parquet_reader.read_row_group(row_group_index, columns=["Address", "Tags"])
            row_group_samples = []
            for address, tags in zip(row_group.column("Address").to_pylist(), row_group.column("Tags").to_pylist()):
                sample = _convert_to_deepparse_tags(address, tags)
                if sample is not None:
                    row_group_samples.append(sample)
            samples = row_group_samples + samples
            if len(samples) >= num_samples:
                break

    return samples[-num_samples:]


def _convert_to_deepparse_tags(address: str, tags: List[str]) -> Tuple[str, List] | None:
    # Map a worldwide-addresses sample to the deepparse tags. Return None if no word is left in the address.
    kept_words = []
    kept_tags = []
    for word, tag in zip(address.split(), tags):
        if tag in WORLDWIDE_ADDRESSES_REMOVED_TAGS:
            continue
        kept_words.append(word)
        kept_tags.append(WORLDWIDE_ADDRESSES_TAGS_MAPPING.get(tag, tag))
    if not kept_words:
        return None
    return " ".join(kept_words), kept_tags


# pylint: disable=pointless-string-statement
FASTTEXT_COPYRIGHT_MIT_LICENSE = """
The code below was copied from the FastText project, and has been modified for the purpose of this package.

COPYRIGHT

All contributions from the https://github.com/facebookresearch/fastText authors.
Copyright (c) 2016 - August 13 2020
All rights reserved.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def download_fasttext_embeddings(cache_dir: str, verbose: bool = True, offline: bool = False) -> str:
    """
    Simpler version of the download_model function from FastText to download pretrained common-crawl
    vectors from FastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
    saving directory (saving_dir).

    Return the full path to the FastText embeddings.
    """
    os.makedirs(cache_dir, exist_ok=True)

    file_name = "cc.fr.300.bin"
    gz_file_name = f"{file_name}.gz"

    file_name_path = os.path.join(cache_dir, file_name)
    if not os.path.isfile(file_name_path) and not offline:
        saving_file_path = os.path.join(cache_dir, gz_file_name)

        download_gz_model(gz_file_name, saving_file_path, verbose=verbose)
        with gzip.open(os.path.join(cache_dir, gz_file_name), "rb") as f:
            with open(os.path.join(cache_dir, file_name), "wb") as f_out:
                shutil.copyfileobj(f, f_out)
        os.remove(os.path.join(cache_dir, gz_file_name))

    return file_name_path  # return the full path to the FastText embeddings


# Now use a saving path and don't return a bool
def download_gz_model(gz_file_name: str, saving_path: str, verbose: bool = True) -> None:
    """
    Simpler version of the _download_gz_model function from FastText to download pretrained common-crawl
    vectors from FastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
    saving directory (saving_path).
    """

    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file_name}"
    if verbose:
        logger.info(
            "The FastText pretrained word embeddings will be downloaded (6.8 GO), "
            "this process will take several minutes."
        )
    _download_file(url, saving_path, verbose=verbose)


# No modification, we just need to call our _print_progress function
def _download_file(url: str, write_file_name: str, chunk_size: int = 2**13, verbose: bool = True) -> None:
    if verbose:
        logger.info("Downloading %s", url)

    response = urlopen(url)  # pylint: disable=consider-using-with
    if hasattr(response, "getheader"):
        file_size = int(response.getheader("Content-Length").strip())
    else:  # pragma: no cover
        file_size = int(response.info().getheader("Content-Length").strip())
    downloaded = 0
    download_file_name = write_file_name + ".part"
    with open(download_file_name, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            if verbose:
                _print_progress(downloaded, file_size)

    os.rename(download_file_name, write_file_name)


# Better print formatting for some shell that don't update properly.
def _print_progress(downloaded_bytes: int, total_size: int) -> None:
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    progress_bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    bar_print = "=" * progress_bar + ">" + " " * (bar_size - progress_bar)
    update = f"\r({percent:0.2f}%) [{bar_print}]"

    sys.stdout.write(update)
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write("\n")


def load_fasttext_embeddings(path: str):
    """
    Function to load fastText ``.bin`` embeddings.

    .. deprecated::
        Use :class:`~deepparse.embeddings_models.FastTextBinReader` instead. This function is kept for backward
        compatibility and returns a ``FastTextBinReader``, which has the same ``model[word]`` and
        ``get_dimension()`` interface as the former ``fasttext`` model object.
    """
    warnings.warn(
        "load_fasttext_embeddings is deprecated and will be removed in a future release, "
        "use deepparse.embeddings_models.FastTextBinReader instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    # Imported here since the embeddings_models package imports this module.
    from .embeddings_models import FastTextBinReader  # pylint: disable=import-outside-toplevel

    return FastTextBinReader(path)
