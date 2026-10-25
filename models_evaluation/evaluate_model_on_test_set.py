"""
Evaluate a pretrained or a retrained model on one or more test pickles, such as the ``test.p`` file built from the
`worldwide-addresses <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_ HuggingFace dataset following
the `deepparse-address-data <https://github.com/GRAAL-Research/deepparse-address-data>`_ protocol.

Unlike ``evaluate_model.py``, it does not need the former per-country test directories. The results are saved in
``models_evaluation/results/<results_type>/test_set_results_<model_type>.json``, with one accuracy per test file
(keyed by the file name without its extension).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def evaluate_on_test_sets(address_parser: AddressParser, test_paths: List[str], batch_size: int) -> Dict[str, float]:
    """
    Compute the test accuracy of a parser on each test pickle.
    """
    results = {}
    for test_path in test_paths:
        test_container = PickleDatasetContainer(test_path)
        test_results = address_parser.test(test_container, batch_size=batch_size, num_workers=4)
        results[Path(test_path).stem] = test_results["test_accuracy"]
        print(f"{test_path}: {results[Path(test_path).stem]:.2f}")
    return results


def main(args: argparse.Namespace) -> None:
    saving_dir = os.path.join(".", "models_evaluation", "results", args.results_type)
    os.makedirs(saving_dir, exist_ok=True)

    address_parser = AddressParser(
        model_type=args.model_type,
        attention_mechanism=args.attention_mechanism,
        device=args.device,
        path_to_retrained_model=args.model_path,
    )
    results = evaluate_on_test_sets(address_parser, args.test_paths, args.batch_size)

    model_name = f"{args.model_type}_attention" if args.attention_mechanism else args.model_type
    with open(os.path.join(saving_dir, f"test_set_results_{model_name}.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, help="Model type to evaluate (e.g. 'fasttext' or 'bpemb').")
    parser.add_argument("test_paths", type=str, nargs="+", help="Path(s) to the test pickle(s), e.g. test.p.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to a retrained model. By default, the pretrained one."
    )
    parser.add_argument("--attention_mechanism", action="store_true", help="Use the attention version of the model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g. 'cuda:0' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size of the data to evaluate on.")
    parser.add_argument(
        "--results_type", type=str, default="actual", help="Subdirectory of models_evaluation/results to save in."
    )

    main(parser.parse_args())
