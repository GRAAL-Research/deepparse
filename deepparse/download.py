import argparse
import os
import warnings

from bpemb import BPEmb

from deepparse import download_fasttext_embeddings, latest_version, download_weights, CACHE_PATH


def main(args: argparse.Namespace) -> None:
    """
    Script to manually download all the dependancies for a pre-trained model.
    """
    model_type = args.model_type
    os.makedirs(CACHE_PATH, exist_ok=True)

    if model_type == "fasttext":
        download_fasttext_embeddings("fr", saving_dir=CACHE_PATH)
    if model_type == "bpemb":
        BPEmb(lang="multi", vs=100000, dim=300)  # The class manage the download of the pre-trained words embedding

    model_path = os.path.join(CACHE_PATH, f"{model_type}.ckpt")
    version_path = os.path.join(CACHE_PATH, f"{model_type}.version")
    if not os.path.isfile(model_path) or not os.path.isfile(version_path):
        download_weights(model_type, CACHE_PATH)
    elif not latest_version(model_type, cache_path=CACHE_PATH):
        warnings.warn("A new version of the pre-trained model is available. The newest model will be downloaded.")
        download_weights(model_type, CACHE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["fasttext", "bpemb"], help="The model type to download.")

    args_parser = parser.parse_args()

    main(args_parser)
