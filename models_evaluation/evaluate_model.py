import argparse
import json
import os

import pycountry

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser
from models_evaluation.tools import clean_up_name, train_country_file, zero_shot_eval_country_file


def main(args):
    address_parser = AddressParser(model_type=args.model_type, device=0)
    directory_path = args.test_directory

    test_files = os.listdir(directory_path)

    training_test_results = {}
    zero_shot_test_results = {}

    saving_dir = os.path.join(".", "results")
    os.makedirs(saving_dir, exist_ok=True)

    for test_file in test_files:
        country = pycountry.countries.get(alpha_2=test_file.replace('.p', '').upper()).name
        country = clean_up_name(country)

        print(f'Testing on test files {country}')

        test_file_path = os.path.join(directory_path, test_file)
        test_container = PickleDatasetContainer(test_file_path)

        results = address_parser.test(test_container, batch_size=4096, num_workers=4,
                                      logging_path=f"./chekpoints/{args.model_type}", checkpoint=args.model_path)

        if train_country_file(test_file):
            training_test_results.update({country: results['test_accuracy']})
        elif zero_shot_eval_country_file(test_file):
            zero_shot_test_results.update({country: results['test_accuracy']})
        else:
            print(f"Error with the identification of test file type {test_file}.")

    json.dump(training_test_results,
              open(os.path.join(saving_dir, f"training_test_results{args.model_type}.json"), "w"))
    json.dump(training_test_results,
              open(os.path.join(saving_dir, f"zero_shot_test_results{args.model_type}.json"), "w"))

    # todo add test on noisy dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_type", type=str, help="Model type to retrain.",
                        choices=["fasttext", "bpemb"])
    parser.add_argument("test_directory", type=str, help="Path to the test directory.")
    parser.add_argument("model_path", type=str, help="Path to the model to evaluate on.")
    args_parser = parser.parse_args()

    main(args_parser)
