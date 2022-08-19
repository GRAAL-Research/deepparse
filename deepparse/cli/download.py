import argparse
import os
import sys

from bpemb import BPEmb

from deepparse import (
    CACHE_PATH,
    download_fasttext_magnitude_embeddings,
    latest_version,
    download_fasttext_embeddings,
    download_weights,
)
from .parser_arguments_adder import choices


def main(args=None) -> None:
    """
    CLI function to manually download all the dependencies for a pretrained model.

    Example of usage:

    .. code-block:: sh

        download_model fasttext

        download_model fasttext a_cache_dir_path
    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]

    parsed_args = get_args(args)

    model_type = parsed_args.model_type
    if "-attention" in model_type:
        root_model_type = model_type.split("-")[0]
        model_type = root_model_type + "_attention"

    saving_cache_path = parsed_args.saving_cache_dir

    if saving_cache_path is None:
        saving_cache_path = CACHE_PATH

    if "fasttext" in model_type and "fasttext-light" not in model_type:
        download_fasttext_embeddings(cache_dir=saving_cache_path)
    elif model_type == "fasttext-light":
        download_fasttext_magnitude_embeddings(cache_dir=saving_cache_path)
    elif "bpemb" in model_type:
        BPEmb(lang="multi", vs=100000, dim=300)  # The class manage the download of the pretrained words embedding

    model_path = os.path.join(saving_cache_path, f"{model_type}.ckpt")
    version_path = os.path.join(saving_cache_path, f"{model_type}.version")
    if not os.path.isfile(model_path) or not os.path.isfile(version_path):
        download_weights(model_type, CACHE_PATH)
    elif not latest_version(model_type, cache_path=saving_cache_path, verbose=True):
        print("A new version of the pretrained model is available. The newest model will be downloaded.")
        download_weights(model_type, saving_cache_path)


def get_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for the cli."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        choices=choices,
        help="The model type to download.",
    )
    parser.add_argument(
        "--saving_cache_dir",
        type=str,
        default=None,
        help="To change the default saving cache directory (default to None e.g. default path).",
    )

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
