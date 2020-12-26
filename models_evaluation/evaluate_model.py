import argparse
import json
import os

from deepparse.parser import AddressParser
from models_evaluation.tools import train_country_file, zero_shot_eval_country_file, test_on_country_data


def main(args):
    saving_dir = os.path.join(".", "models_evaluation", "results")
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

    json.dump(training_test_results,
              open(os.path.join(saving_dir, f"training_test_results_{args.model_type}.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(zero_shot_test_results,
              open(os.path.join(saving_dir, f"zero_shot_test_results_{args.model_type}.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    noisy_test_directory = args.noisy_test_directory
    noisy_test_files = os.listdir(noisy_test_directory)
    noisy_training_test_results = {}
    for idx, noisy_test_file in enumerate(noisy_test_files):
        results, country = test_on_country_data(address_parser, noisy_test_file, directory_path, args)
        print(f"{idx} file done of {len(noisy_test_files)}.")

        if train_country_file(noisy_test_file):
            noisy_training_test_results.update({country: results['test_accuracy']})
        else:
            print(f"Error with the identification of test file type {noisy_test_file}.")

    json.dump(noisy_training_test_results,
              open(os.path.join(saving_dir, f"training_noisy_test_results_{args.model_type}.json"),
                   "w",
                   encoding="utf-8"),
              ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_type", type=str, help="Model type to retrain.", choices=["fasttext", "bpemb"])
    parser.add_argument("test_directory", type=str, help="Path to the test directory.")
    parser.add_argument("noisy_test_directory", type=str, help="Path the to noisy test directory.")
    parser.add_argument("model_path", type=str, help="Path to the model to evaluate on.")
    args_parser = parser.parse_args()

    main(args_parser)
