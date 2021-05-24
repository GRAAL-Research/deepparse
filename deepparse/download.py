import argparse
import os

from bpemb import BPEmb

from deepparse import CACHE_PATH, download_fasttext_magnitude_embeddings, latest_version, \
    download_fasttext_embeddings, download_weights


def main(args: argparse.Namespace) -> None:
    """
    Script to manually download all the dependencies for a pre-trained model.
    """
    model_type = args.model_type

    if model_type == "fasttext":
        download_fasttext_embeddings(saving_dir=CACHE_PATH)
    if model_type == "fasttext-light":
        download_fasttext_magnitude_embeddings(saving_dir=CACHE_PATH)
    if model_type == "bpemb":
        BPEmb(lang="multi", vs=100000, dim=300)  # The class manage the download of the pre-trained words embedding

    model_path = os.path.join(CACHE_PATH, f"{model_type}.ckpt")
    version_path = os.path.join(CACHE_PATH, f"{model_type}.version")
    if not os.path.isfile(model_path) or not os.path.isfile(version_path):
        download_weights(model_type, CACHE_PATH)
    elif not latest_version(model_type, cache_path=CACHE_PATH):
        print("A new version of the pre-trained model is available. The newest model will be downloaded.")
        download_weights(model_type, CACHE_PATH)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type",
                        choices=["fasttext", "fasttext-light", "bpemb"],
                        help="The model type to download.")

    args_parser = parser.parse_args()

    main(args_parser)
