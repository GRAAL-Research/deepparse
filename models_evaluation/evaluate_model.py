# pylint: disable=too-many-locals

import argparse
import json
import os

from deepparse.parser import AddressParser
from models_evaluation.tools import train_country_file, zero_shot_eval_country_file, test_on_country_data


def main(args):
    results_type = args.results_type
    saving_dir = os.path.join(".", "models_evaluation", "results", results_type)
    os.makedirs(saving_dir, exist_ok=True)

    address_parser = AddressParser(model_type=args.model_type, device=0)
    directory_path = args.test_directory
    test_files = os.listdir(directory_path)
    training_test_results = {}
    zero_shot_test_results = {}
    for idx, test_file in enumerate(test_files):
        results, country = test_on_country_data(address_parser, test_file, directory_path, args)
        print(f"{idx} file done of {len(test_files)}.")

        if train_country_file(test_file):
            training_test_results.update({country: results['test_accuracy']})
        elif zero_shot_eval_country_file(test_file):
            zero_shot_test_results.update({country: results['test_accuracy']})
        else:
            print(f"Error with the identification of test file type {test_file}.")

    training_base_string = "training_test_results"
    training_incomplete_base_string = "training_incomplete_test_results"
    zero_shot_base_string = "zero_shot_test_results"

    with open(os.path.join(saving_dir, f"{training_base_string}_{args.model_type}.json"), "w",
              encoding="utf-8") as file:
        json.dump(training_test_results, file, ensure_ascii=False)

    with open(os.path.join(saving_dir, f"{zero_shot_base_string}_{args.model_type}.json"), "w",
              encoding="utf-8") as file:
        json.dump(zero_shot_test_results, file, ensure_ascii=False)

    incomplete_test_directory = args.incomplete_test_directory
    incomplete_test_files = os.listdir(incomplete_test_directory)
    incomplete_training_test_results = {}
    for idx, incomplete_test_file in enumerate(incomplete_test_files):
        results, country = test_on_country_data(address_parser, incomplete_test_file, incomplete_test_directory, args)
        print(f"{idx} file done of {len(incomplete_test_files)}.")

        if train_country_file(incomplete_test_file):
            incomplete_training_test_results.update({country: results['test_accuracy']})
        else:
            print(f"Error with the identification of test file type {incomplete_test_file}.")

    with open(os.path.join(saving_dir, f"{training_incomplete_base_string}_{args.model_type}.json"),
              "w",
              encoding="utf-8") as file:
        json.dump(incomplete_training_test_results, file, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_type", type=str, help="Model type to retrain.", choices=["fasttext", "bpemb"])
    parser.add_argument("test_directory", type=str, help="Path to the test directory.")
    parser.add_argument("incomplete_test_directory", type=str, help="Path the to incomplete test directory.")
    parser.add_argument("model_path", type=str, help="Path to the model to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size of the data to evaluate on.")
    parser.add_argument("--results_type",
                        type=str,
                        default="actual",
                        help="Either or not the evaluation is for new models.")
    args_parser = parser.parse_args()

    main(args_parser)
