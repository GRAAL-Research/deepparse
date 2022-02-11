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


def main(args=None) -> None:
    """
    cli function to manually download all the dependencies for a pre-trained model.

    Example of usage:

    .. code-block:: sh

        download_model fasttext
    """
    if args is None:
        args = sys.argv[1:]

    parsed_args = get_args(args)

    model_type = parsed_args.model_type

    if "fasttext" in model_type and "fasttext-light" not in model_type:
        download_fasttext_embeddings(saving_dir=CACHE_PATH)
    elif model_type == "fasttext-light":
        download_fasttext_magnitude_embeddings(saving_dir=CACHE_PATH)
    elif "bpemb" in model_type:
        BPEmb(lang="multi", vs=100000, dim=300)  # The class manage the download of the pre-trained words embedding

    model_path = os.path.join(CACHE_PATH, f"{model_type}.ckpt")
    version_path = os.path.join(CACHE_PATH, f"{model_type}.version")
    if not os.path.isfile(model_path) or not os.path.isfile(version_path):
        download_weights(model_type, CACHE_PATH)
    elif not latest_version(model_type, cache_path=CACHE_PATH):
        print("A new version of the pre-trained model is available. The newest model will be downloaded.")
        download_weights(model_type, CACHE_PATH)


def get_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for the cli."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        choices=[
            "fasttext",
            "fasttext-attention",
            "fasttext-light",
            "bpemb",
            "bpemb-attention",
        ],
        help="The model type to download.",
    )

    return parser


def get_args(args):  # pragma: no cover
    """Parse arguments passed in from shell."""
    return get_parser().parse_args(args)
