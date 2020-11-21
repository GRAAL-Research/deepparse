import argparse
import os
import warnings

from bpemb import BPEmb

from deepparse import download_fasttext_embeddings, verify_latest_version, download_weights, CACHE_PATH


def main(args: argparse.Namespace) -> None:
    """
    Script to manually download all the dependancies for a pre-trained model.
    """
    model = args.model
    os.makedirs(CACHE_PATH, exist_ok=True)

    if model == "fasttext":
        download_fasttext_embeddings("fr", saving_dir=CACHE_PATH)
    if model == "bpemb":
        BPEmb(lang="multi", vs=100000, dim=300)  # The class manage the download of the pre-trained words embedding

    model_path = os.path.join(CACHE_PATH, f"{model}.ckpt")
    version_path = os.path.join(CACHE_PATH, f"{model}.version")
    if not os.path.isfile(model_path) or not os.path.isfile(version_path):
        download_weights(model, CACHE_PATH)
    elif verify_latest_version(model):
        warnings.warn("A new version of the pre-trained model is available. The newest model will be downloaded.")
        download_weights(model, CACHE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["fasttext", "bpemb"], help="The model type to download.")

    args_parser = parser.parse_args()

    main(args_parser)
