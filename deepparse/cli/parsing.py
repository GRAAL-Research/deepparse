import argparse
import os

from bpemb import BPEmb

from deepparse import (
    CACHE_PATH,
    download_fasttext_magnitude_embeddings,
    latest_version,
    download_fasttext_embeddings,
    download_weights,
)


def main(args: argparse.Namespace) -> None:
    """
    CLI function to rapidly parse an address dataset and output it in another file.
    """
    parsing_model = args.parsing_model




if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "parsing_model",
        choices=[
            "fasttext",
            "fasttext_attention",
            "fasttext-light",
            "bpemb",
            "bpemb_attention",
        ],
        help="The parsing module to use.",
    )

    parser.add_argument(
        "dataset_path",
        help="The path to the dataset file in a pickle or CSV format.",
    )

    parser.add_argument(
        "dataset_path",
        help="The path to the dataset file in a pickle or CSV format.",
    )

    # arg pour le csv data container
    # output file?

    args_parser = parser.parse_args()

    main(args_parser)
