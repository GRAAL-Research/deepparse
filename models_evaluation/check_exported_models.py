"""
Check the exported models end to end before pushing them to the Hugging Face Hub, without any network access.

We build a fake Hugging Face cache holding the exported repositories (``export_model_to_hub.py``) and the Model2Vec
embeddings of a real cache, then, for each selected model:

- load it as a user would (``AddressParser(model_type=..., offline=True)``) and check that it is not randomly
  initialized, has the selected hidden size and gives the same predictions as its retrained checkpoint on a sample of
  the test addresses;
- retrain it for one epoch without ``seq2seq_params`` and reload the retrained checkpoint.

Usage:
    python -m models_evaluation.check_exported_models selection.json hub_export cache data/test.p
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List

from deepparse.dataset_container import ListDatasetContainer
from deepparse.parser import AddressParser
from models_evaluation.export_model_to_hub import repo_id_of


def build_fake_hub_cache(selection: List[Dict], export_dir: Path, embeddings_cache: Path, cache_dir: Path) -> None:
    """
    Lay the exported repositories out as a Hugging Face cache (``models--<org>--<name>/snapshots/<commit>`` and
    ``refs/main``) and link the Model2Vec embeddings of ``embeddings_cache``.
    """
    for entry in embeddings_cache.iterdir():
        if entry.name.startswith("models--minishlab--"):
            os.symlink(entry.resolve(), cache_dir / entry.name)
    for selected in selection:
        repo_id = repo_id_of(selected["model_name"])
        commit = hashlib.sha1(repo_id.encode("utf-8")).hexdigest()
        repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")
        shutil.copytree(export_dir / repo_id.split("/")[1], repo_dir / "snapshots" / commit)


def check_model(  # pylint: disable=too-many-locals
    selected: Dict, cache_dir: Path, embeddings_cache: Path, sample: List, work_dir: Path
) -> List[str]:
    """
    Return the problems found with one exported model (an empty list if none).
    """
    run = selected["selected_run"]
    parser_kwargs = {
        "model_type": run["model_type"],
        "attention_mechanism": run["attention_mechanism"],
        "device": "cpu",
        "offline": True,
        "verbose": False,
    }
    problems = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        address_parser = AddressParser(cache_dir=str(cache_dir), **parser_kwargs)
    if any("randomly initialized" in str(warning.message) for warning in caught):
        problems.append("randomly initialized instead of loading the exported weights")
    if address_parser.model.encoder.lstm.hidden_size != run["hidden_size"]:
        problems.append(f"hidden size {address_parser.model.encoder.lstm.hidden_size} instead of {run['hidden_size']}")

    reference = AddressParser(
        cache_dir=str(embeddings_cache), path_to_retrained_model=run["checkpoint"], **parser_kwargs
    )
    addresses = [address for address, _ in sample]
    predictions = [parsed.address_parsed_components for parsed in address_parser(addresses, batch_size=256)]
    expected = [parsed.address_parsed_components for parsed in reference(addresses, batch_size=256)]
    differences = sum(prediction != expectation for prediction, expectation in zip(predictions, expected))
    if differences:
        problems.append(f"{differences}/{len(addresses)} addresses parsed differently than the retrained checkpoint")

    logging_path = work_dir / selected["model_name"]
    address_parser.retrain(
        ListDatasetContainer(sample[:300], is_training_container=True),
        train_ratio=0.8,
        epochs=1,
        batch_size=64,
        num_workers=0,
        logging_path=str(logging_path),
        verbose=False,
    )
    retrained_checkpoint = next(logging_path.glob("retrained_*_address_parser.ckpt"))
    try:
        AddressParser(cache_dir=str(cache_dir), path_to_retrained_model=str(retrained_checkpoint), **parser_kwargs)
    except RuntimeError as error:
        problems.append(f"the retrained model does not reload: {str(error).splitlines()[0]}")
    return problems


def main(args: argparse.Namespace) -> None:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    with open(args.test_path, "rb") as file:
        test_data = pickle.load(file)
    random.seed(0)
    sample = random.sample(test_data, args.num_addresses)

    with tempfile.TemporaryDirectory() as temp_dir:
        # The cache is kept with --keep_cache, e.g. to measure the speed of the models before the push.
        cache_dir = Path(args.keep_cache) if args.keep_cache else Path(temp_dir) / "cache"
        work_dir = Path(temp_dir) / "work"
        cache_dir.mkdir(parents=True)
        build_fake_hub_cache(selection, Path(args.export_dir), Path(args.cache_dir), cache_dir)
        failed = False
        for selected in selection:
            problems = check_model(selected, cache_dir, Path(args.cache_dir), sample, work_dir)
            failed = failed or bool(problems)
            print(f"{selected['model_name']:35s} {'OK' if not problems else 'FAILED: ' + '; '.join(problems)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selection", type=str, help="The selection JSON file written by select_best_model.")
    parser.add_argument("export_dir", type=str, help="The output directory of export_model_to_hub.")
    parser.add_argument("cache_dir", type=str, help="A deepparse cache holding the Model2Vec embeddings.")
    parser.add_argument("test_path", type=str, help="The test pickle to sample the addresses from.")
    parser.add_argument("--num_addresses", type=int, default=2000)
    parser.add_argument("--keep_cache", type=str, default=None, help="Build the fake Hub cache in this new directory.")
    main(parser.parse_args())
