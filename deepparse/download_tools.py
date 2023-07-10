import gzip
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Union
from urllib.request import urlopen

import requests
from bpemb import BPEmb
from fasttext.FastText import _FastText
from requests import HTTPError
from urllib3.exceptions import MaxRetryError

from .errors.server_error import ServerError

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


def download_fasttext_magnitude_embeddings(cache_dir: str, verbose: bool = True, offline: bool = False) -> str:
    """
    Function to download the magnitude pretrained fastText model.

    Return the full path to the fastText embeddings.
    """

    os.makedirs(cache_dir, exist_ok=True)

    model = "fasttext"
    extension = "magnitude"
    file_name = os.path.join(cache_dir, f"{model}.{extension}")
    if not os.path.isfile(file_name) and not offline:
        if verbose:
            print(
                "The fastText pretrained word embeddings will be download in magnitude format (2.3 GO), "
                "this process will take several minutes."
            )
        extension = extension + ".gz"
        download_from_public_repository(file_name=model, saving_dir=cache_dir, file_extension=extension)
        gz_file_name = file_name + ".gz"
        with gzip.open(os.path.join(cache_dir, gz_file_name), "rb") as f:
            with open(os.path.join(cache_dir, file_name), "wb") as f_out:
                shutil.copyfileobj(f, f_out)
        os.remove(os.path.join(cache_dir, gz_file_name))
    return file_name


def download_weights(model_filename: str, saving_dir: str, verbose: bool = True) -> None:
    """
    Function to download the pretrained weights of one of our pre-trained base models.
    Args:
       model_filename: The network type (i.e. ``fasttext`` or ``bpemb``).
        saving_dir: The path to the saving directory.
        verbose (bool): Either or not to be verbose during the download of a model. The default value is True.
    """
    if verbose:
        print(f"Downloading the pre-trained weights for the network {model_filename}.")

    try:
        download_from_public_repository(model_filename, saving_dir, "ckpt")
        download_from_public_repository(model_filename, saving_dir, "version")
    except requests.exceptions.ConnectTimeout as error:
        raise ServerError(
            "There was an error trying to connect to the Deepparse server. Please try again later."
        ) from error


def download_from_public_repository(file_name: str, saving_dir: str, file_extension: str) -> None:
    """
    Simple function to download the content of a file from Deepparse public repository.
    The repository URL string is `'https://graal.ift.ulaval.ca/public/deepparse/{}.{}'``
    where the first bracket is the file name and the second is the file extension.
    """
    url = BASE_URL.format(file_name, file_extension)
    r = requests.get(url, timeout=5)
    r.raise_for_status()  # Raise exception
    Path(saving_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(saving_dir, f"{file_name}.{file_extension}"), "wb") as file:
        file.write(r.content)


def download_models(saving_cache_path: Union[Path, None] = None) -> None:
    """
    Function to download all the pretrained models.  It will download all the models checkpoint and version file.

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
        BPEmb(
            lang="multi", vs=100000, dim=300, cache_dir=saving_cache_path
        )  # The class manage the download of the pretrained words embedding

    model_type_filename = MODEL_MAPPING_CHOICES[model_type]
    model_path = os.path.join(saving_cache_path, f"{model_type_filename}.ckpt")
    version_path = os.path.join(saving_cache_path, f"{model_type_filename}.version")
    if not os.path.isfile(model_path) or not os.path.isfile(version_path):
        download_weights(model_type_filename, saving_dir=saving_cache_path)
    elif not latest_version(model_type_filename, cache_path=saving_cache_path, verbose=True):
        print("A new version of the pretrained model is available. The newest model will be downloaded.")
        download_weights(model_type_filename, saving_dir=saving_cache_path)


def latest_version(model: str, cache_path: str, verbose: bool) -> bool:
    """
    Verify if the local model is the latest.
    """
    # Reading of the actual local version
    with open(os.path.join(cache_path, model + ".version"), encoding="utf-8") as local_model_hash_file:
        local_model_hash_version = local_model_hash_file.readline()

    # We create a temporary directory for the server-side version file
    tmp_cache = os.path.join(cache_path, "tmp")
    try:
        # We create a temporary directory for the server-side version file
        os.makedirs(tmp_cache, exist_ok=True)

        download_from_public_repository(model, tmp_cache, "version")

        # Reading of the server-side version
        with open(os.path.join(tmp_cache, model + ".version"), encoding="utf-8") as remote_model_hash_file:
            remote_model_hash_version = remote_model_hash_file.readline()

        is_latest_version = local_model_hash_version.strip() == remote_model_hash_version.strip()

    except HTTPError as exception:  # HTTP connection error handling
        if HTTP_CLIENT_ERROR_STATUS_CODE <= exception.response.status_code < NEXT_RANGE_STATUS_CODE:
            # Case where Deepparse server is down.
            if verbose:
                warnings.warn(
                    f"We where not able to verify the cached model in the cache directory {cache_path}. It seems like"
                    f"Deepparse server is not available at the moment. We recommend to attempt to verify "
                    f"the model version another time using our download CLI function.",
                    category=RuntimeWarning,
                )
            # The is_lastest_version is set to True even if we were not able to validate the version. We do so not to
            # block the rest of the process.
            is_latest_version = True
        else:
            # We re-raise the exception if the status_code is not in the two ranges we are interested in
            # (local server or remote server error).
            raise
    except MaxRetryError:
        # Case where the user does not have an Internet connection. For example, one can run it in a
        # Docker container not connected to the Internet.
        if verbose:
            warnings.warn(
                f"We where not able to verify the cached model in the cache directory {cache_path}. It seems like"
                f"you are not connected to the Internet. We recommend to verify if you have the latest using our "
                f"download CLI function.",
                category=RuntimeWarning,
            )
        # The is_lastest_version is set to True even if we were not able to validate the version. We do so not to
        # block the rest of the process.
        is_latest_version = True
    finally:
        # Cleaning the temporary directory
        if os.path.exists(tmp_cache):
            shutil.rmtree(tmp_cache)

    return is_latest_version


# pylint: disable=pointless-string-statement
FASTTEXT_COPYRIGHT_MIT_LICENSE = """
The code below was copied from the fastText project, and has been modified for the purpose of this package.

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
    Simpler version of the download_model function from fastText to download pretrained common-crawl
    vectors from fastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
    saving directory (saving_dir).

    Return the full path to the fastText embeddings.
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

    return file_name_path  # return the full path to the fastText embeddings


# Now use a saving path and don't return a bool
def download_gz_model(gz_file_name: str, saving_path: str, verbose: bool = True) -> None:
    """
    Simpler version of the _download_gz_model function from fastText to download pretrained common-crawl
    vectors from fastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
    saving directory (saving_path).
    """

    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file_name}"
    if verbose:
        print(
            "The fastText pretrained word embeddings will be downloaded (6.8 GO), "
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
