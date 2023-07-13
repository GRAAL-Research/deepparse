import argparse
import sys

from deepparse.download_tools import download_models


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

    download_models(saving_cache_path=saving_cache_path)


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
