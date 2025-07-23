import gzip
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Union
from urllib.request import urlopen

from fasttext.FastText import _FastText
from huggingface_hub import hf_hub_download, snapshot_download
from transformers.utils.hub import cached_file, extract_commit_hash
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from .bpemb_url_bug_fix import BPEmbBaseURLWrapperBugFix

BASE_URL = "https://graal.ift.ulaval.ca/public/deepparse/{}.{}"
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
}

MODEL_REPO_IDS = {
    "bpemb": "deepparse/bpemb-base",
    "bpemb_attention": "deepparse/bpemb-attention",
    "fasttext": "deepparse/fasttext-base",
    "fasttext_attention": "deepparse/fasttext-attention",
    "fasttext-light": "deepparse/fasttext-base",
    "fasttext-light_attention": "deepparse/fasttext-attention",
}


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
            print(
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
                f"The offline parameter is set to False, so if a new pre-trained {model_type} model is available it will "
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


def download_models(saving_cache_path: Union[Path, None] = None) -> None:
    """
    Function to download all the pretrained models.  It will download all the model's checkpoints and version files.

    Args:
        saving_cache_path: The path to the saving cache directory for the specified model.
    """
    for model_type in MODEL_MAPPING_CHOICES:
        download_model(model_type, saving_cache_path=saving_cache_path)


def download_model(
    model_type: str,
    saving_cache_path: Union[Path, None] = None,
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

    model_type = MODEL_MAPPING_CHOICES[model_type]

    download_weights(model_type, saving_cache_path, verbose=True, offline=False)


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
        print(
            "The FastText pretrained word embeddings will be downloaded (6.8 GO), "
            "this process will take several minutes."
        )
    _download_file(url, saving_path, verbose=verbose)


# No modification, we just need to call our _print_progress function
def _download_file(url: str, write_file_name: str, chunk_size: int = 2**13, verbose: bool = True) -> None:
    if verbose:
        print(f"Downloading {url}")

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
    update = f"\r(%0.2f%%) [{bar_print}]" % percent

    sys.stdout.write(update)
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write("\n")


# The difference with the original code is the removal of the print warning.
def load_fasttext_embeddings(path: str) -> _FastText:
    """
    Wrapper to load a model given a filepath and return a model object.
    """
    return _FastText(model_path=path)
