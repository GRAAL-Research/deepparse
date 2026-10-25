# pylint: disable=consider-using-with

import os
import pickle
import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock

from deepparse.dataset_container import PickleDatasetContainer
from models_evaluation.evaluate_model_on_test_set import evaluate_on_test_sets
from models_evaluation.tools import test_on_country_data as evaluate_on_country_data

A_DATASET = [
    ("350 rue des lilas ouest quebec g1l 1b6", ["StreetNumber", "StreetName", "StreetName", "StreetName", "StreetName",
                                                 "Municipality", "PostalCode", "PostalCode"]),
]  # fmt: skip


class EvaluateModelOnTestSetTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.test_paths = []
        for name in ("test", "ca"):
            path = os.path.join(self.temp_dir_obj.name, f"{name}.p")
            with open(path, "wb") as file:
                pickle.dump(A_DATASET, file)
            self.test_paths.append(path)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_givenTestPickles_whenEvaluate_thenReturnTheAccuracyOfEachFileKeyedByItsName(self):
        address_parser = MagicMock()
        address_parser.test.side_effect = [{"test_accuracy": 97.5}, {"test_accuracy": 88.0}]

        actual = evaluate_on_test_sets(address_parser, self.test_paths, batch_size=16)

        self.assertEqual({"test": 97.5, "ca": 88.0}, actual)
        test_container = address_parser.test.call_args_list[0].args[0]
        self.assertIsInstance(test_container, PickleDatasetContainer)
        self.assertEqual(16, address_parser.test.call_args_list[0].kwargs["batch_size"])

    def test_givenACountryTestFile_whenTestOnCountryData_thenCallTestWithItsSupportedArguments(self):
        # The former call passed logging_path and checkpoint, which AddressParser.test no longer accepts.
        address_parser = MagicMock()
        address_parser.test.return_value = {"test_accuracy": 90.0}

        results, country = evaluate_on_country_data(
            address_parser, "ca.p", self.temp_dir_obj.name, Namespace(batch_size=8, model_type="fasttext")
        )

        self.assertEqual({"test_accuracy": 90.0}, results)
        self.assertEqual("Canada", country)
        self.assertEqual({"batch_size": 8, "num_workers": 4}, address_parser.test.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
