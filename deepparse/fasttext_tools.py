import gzip
import os
import shutil
import sys
from urllib.request import urlopen

from fasttext.FastText import _FastText

from .tools import download_from_url


def download_fasttext_magnitude_embeddings(saving_dir: str, verbose: bool = True) -> str:
    """
    Function to download the magnitude pre-trained fastText model.
    """
    os.makedirs(saving_dir, exist_ok=True)

    model = "fasttext"
    extension = "magnitude"
    file_name = os.path.join(saving_dir, f"{model}.{extension}")
    if not os.path.isfile(file_name):
        if verbose:
            print("The fastText pre-trained word embeddings will be download in magnitude format (2.3 GO), "
                  "this process will take several minutes.")
        extension = extension + ".gz"
        download_from_url(file_name=model, saving_dir=saving_dir, file_extension=extension)
        gz_file_name = file_name + ".gz"
        with gzip.open(os.path.join(saving_dir, gz_file_name), "rb") as f:
            with open(os.path.join(saving_dir, file_name), "wb") as f_out:
                shutil.copyfileobj(f, f_out)
        os.remove(os.path.join(saving_dir, gz_file_name))
    return file_name


# pylint: disable=pointless-string-statement
"""
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


def download_fasttext_embeddings(saving_dir: str, verbose: bool = True) -> str:
    """
        Simpler version of the download_model function from fastText to download pre-trained common-crawl
        vectors from fastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
        saving directory (saving_dir).
    """
    os.makedirs(saving_dir, exist_ok=True)

    file_name = "cc.fr.300.bin"
    gz_file_name = "%s.gz" % file_name

    file_name_path = os.path.join(saving_dir, file_name)
    if os.path.isfile(file_name_path):
        return file_name_path  # return the full path to the fastText embeddings

    saving_file_path = os.path.join(saving_dir, gz_file_name)

    download_gz_model(gz_file_name, saving_file_path, verbose=verbose)
    with gzip.open(os.path.join(saving_dir, gz_file_name), "rb") as f:
        with open(os.path.join(saving_dir, file_name), "wb") as f_out:
            shutil.copyfileobj(f, f_out)
    os.remove(os.path.join(saving_dir, gz_file_name))

    return file_name_path  # return the full path to the fastText embeddings


# Now use a saving path and don't return a bool
def download_gz_model(gz_file_name: str, saving_path: str, verbose: bool = True) -> None:
    """
    Simpler version of the _download_gz_model function from fastText to download pre-trained common-crawl
    vectors from fastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
    saving directory (saving_path).
    """

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/%s" % gz_file_name
    if verbose:
        print("The fastText pre-trained word embeddings will be downloaded (6.8 GO), "
              "this process will take several minutes.")
    _download_file(url, saving_path, verbose=verbose)


# No modification, we just need to call our _print_progress function
def _download_file(url: str, write_file_name: str, chunk_size: int = 2**13, verbose: bool = True) -> None:
    if verbose:
        print("Downloading %s" % url)
    response = urlopen(url)
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
def _print_progress(downloaded_bytes, total_size):
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
def load_fasttext_embeddings(path):
    """
    Load a model given a filepath and return a model object.
    """
    return _FastText(model_path=path)
