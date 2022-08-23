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

models_type = ["fasttext", "fasttext_attention", "bpemb", "bpemb_attention"]


def main(args=None) -> None:
    """
    CLI function to manually download all the dependencies for all pretrained models.

    Example of usage:

    .. code-block:: sh

        download_models

        download_models --saving_cache_dir a_cache_dir_path
    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]

    parsed_args = get_args(args)

    saving_cache_path = parsed_args.saving_cache_dir

    if saving_cache_path is None:
        saving_cache_path = CACHE_PATH

    download_fasttext_embeddings(cache_dir=saving_cache_path)
    download_fasttext_magnitude_embeddings(cache_dir=saving_cache_path)
    BPEmb(
        lang="multi", vs=100000, dim=300, cache_dir=saving_cache_path
    )  # The class manage the download of the pretrained words embedding

    for model_type in models_type:
        model_path = os.path.join(saving_cache_path, f"{model_type}.ckpt")
        version_path = os.path.join(saving_cache_path, f"{model_type}.version")
        if not os.path.isfile(model_path) or not os.path.isfile(version_path):
            download_weights(model_type, saving_dir=saving_cache_path)
        elif not latest_version(model_type, cache_path=saving_cache_path, verbose=True):
            print("A new version of the pretrained model is available. The newest model will be downloaded.")
            download_weights(model_type, saving_dir=saving_cache_path)


def get_parser() -> argparse.ArgumentParser:
    """Return ArgumentParser for the cli."""
    parser = argparse.ArgumentParser()

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
