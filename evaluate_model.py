# A script to evaluate our model before releasing it in order to create a results table
import argparse
import json
import os

import pycountry

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser

# country that we trained on
train_test_files = ['br.p', 'us.p', 'kp.p', 'ru.p', 'de.p', 'fr.p', 'nl.p', 'ch.p', 'fi.p', 'es.p',
                    'cz.p', 'gb.p', 'mx.p', 'no.p', 'ca.p', 'it.p', 'au.p', 'dk.p', 'pl.p', 'at.p']

# country that we did not train on
other_test_files = ['ie.p', 'rs.p', 'uz.p', 'ua.p', 'za.p', 'py.p', 'gr.p', 'dz.p', 'by.p', 'se.p', 'pt.p', 'hu.p',
                    'is.p', 'co.p', 'lv.p', 'my.p', 'ba.p', 'in.p', 're.p',
                    'hr.p', 'ee.p', 'nc.p', 'jp.p', 'nz.p', 'sg.p', 'ro.p', 'bd.p', 'sk.p', 'ar.p', 'kz.p', 've.p',
                    'id.p', 'bg.p', 'cy.p', 'bm.p', 'md.p', 'si.p', 'lt.p',
                    'ph.p', 'be.p', 'fo.p']


def clean_up_name(country):
    """
    Function to clean up pycountry name
    """
    if "Korea" in country:
        country = "South Korea"
    elif "Russian Federation" in country:
        country = "Russia"
    elif "Venezuela" in country:
        country = "Venezuela"
    elif "Moldova" in country:
        country = "Moldova"
    elif "Bosnia" in country:
        country = "Bosnia"
    return country


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

        if test_file in train_test_files:
            training_test_results.update({country: results['test_accuracy']})
        elif test_file in other_test_files:
            zero_shot_test_results.update({country: results['test_accuracy']})

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
