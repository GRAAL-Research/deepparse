# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, too-many-arguments

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from pickle import dump
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd


class RetrainTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_data_saving_dir = os.path.join(cls.temp_dir_obj.name, "data")
        os.makedirs(cls.a_data_saving_dir, exist_ok=True)

        data = [
            (
                '350 rue des Lilas Ouest Quebec city Quebec G1L 1B6',
                [
                    'StreetNumber',
                    'StreetName',
                    'StreetName',
                    'StreetName',
                    'Municipality',
                    'Municipality',
                    'Municipality',
                    'Province',
                    'PostalCode',
                    'PostalCode',
                ],
            ),
            (
                '350 rue des Lilas Ouest Quebec city Quebec G1L 1B6',
                [
                    'StreetNumber',
                    'StreetName',
                    'StreetName',
                    'StreetName',
                    'Municipality',
                    'Municipality',
                    'Municipality',
                    'Province',
                    'PostalCode',
                    'PostalCode',
                ],
            ),
        ]

        training_dataset_name = "train_sample_data"
        test_dataset_name = "test_sample_data"

        cls.a_train_pickle_dataset_path = os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + "p")
        dump(data, open(cls.a_train_pickle_dataset_path, "wb"))

        cls.a_test_pickle_dataset_path = os.path.join(cls.a_data_saving_dir, test_dataset_name + "." + "p")
        dump(data, open(cls.a_test_pickle_dataset_path, "wb"))

        df = pd.DataFrame(data, columns=["Address", "Tags"])
        cls.a_train_csv_dataset_path = os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + "csv")
        df.to_csv(cls.a_train_csv_dataset_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()
