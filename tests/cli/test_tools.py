# pylint: disable=too-many-public-methods

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import argparse
import json
import os
import pickle
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from deepparse.cli import (
    is_csv_path,
    is_pickle_path,
    to_csv,
    to_pickle,
    generate_export_path,
    is_json_path,
    to_json,
    replace_path_extension,
    attention_model_type_handling,
    bool_parse,
    data_container_factory,
)
from deepparse.parser import FormattedParsedAddress


class ToolsTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.an_export_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        self.sep = "\t"

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_givenCSVPath_whenCSVPath_returnTrue(self):
        a_csv_path = "a_path.csv"

        self.assertTrue(is_csv_path(a_csv_path))

        a_csv_path = "a/path/a_path.csv"

        self.assertTrue(is_csv_path(a_csv_path))

        a_csv_path = "./relative/path/a_path.csv"

        self.assertTrue(is_csv_path(a_csv_path))

    def test_givenNotACSVPath_whenCSVPath_returnFalse(self):
        not_a_csv_path = "a_path.tsv"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.doc"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.txt"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.p"

        self.assertFalse(is_csv_path(not_a_csv_path))

        not_a_csv_path = "a_path.pickle"

        self.assertFalse(is_csv_path(not_a_csv_path))

    def test_givenPicklePath_whenPicklePath_returnTrue(self):
        a_pickle_path = "a_path.pickle"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "a/path/a_path.pickle"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "./relative/path/a_path.pickle"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "a_path.p"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "a/path/a_path.p"

        self.assertTrue(is_pickle_path(a_pickle_path))

        a_pickle_path = "./relative/path/a_path.p"

        self.assertTrue(is_pickle_path(a_pickle_path))

    def test_givenNotAPicklePath_whenPicklePath_returnFalse(self):
        not_a_pickle_path = "a_path.tsv"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.doc"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.txt"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.csv"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

        not_a_pickle_path = "a_path.md"

        self.assertFalse(is_pickle_path(not_a_pickle_path))

    def test_givenJSONPath_whenJSONPath_returnTrue(self):
        a_json_path = "a_path.json"

        self.assertTrue(is_json_path(a_json_path))

        a_json_path = "a/path/a_path.json"

        self.assertTrue(is_json_path(a_json_path))

        a_json_path = "./relative/path/a_path.json"

        self.assertTrue(is_json_path(a_json_path))

    def test_givenNotAJSONPath_whenJSONPath_returnFalse(self):
        not_a_json_path = "a_path.tsv"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.doc"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.txt"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.csv"

        self.assertFalse(is_json_path(not_a_json_path))

        not_a_json_path = "a_path.md"

        self.assertFalse(is_json_path(not_a_json_path))

    def test_integration_list_formatted_addresses_to_csv(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_list_of_parsed_addresses = [
            FormattedParsedAddress({a_address_str: a_parsed_address}),
            FormattedParsedAddress({a_address_str: a_parsed_address}),
        ]

        to_csv(a_list_of_parsed_addresses, export_path=self.an_export_path, sep=self.sep)

        parsed_data = pd.read_csv(self.an_export_path, sep=self.sep)
        self.assertEqual(parsed_data.Address[0], a_address_str)
        self.assertEqual(parsed_data.Address[1], a_address_str)

    def test_integration_formatted_address_to_csv(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_parsed_address = FormattedParsedAddress({a_address_str: a_parsed_address})

        to_csv(a_parsed_address, export_path=self.an_export_path, sep=self.sep)

        parsed_data = pd.read_csv(self.an_export_path, sep=self.sep)
        self.assertEqual(parsed_data.Address[0], a_address_str)

    def test_integration_list_formatted_addresses_to_pickle(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_list_of_parsed_addresses = [
            FormattedParsedAddress({a_address_str: a_parsed_address}),
            FormattedParsedAddress({a_address_str: a_parsed_address}),
        ]

        to_pickle(a_list_of_parsed_addresses, export_path=self.an_export_path)

        with open(self.an_export_path, "rb") as file:
            parsed_data = pickle.load(file)
        self.assertEqual(parsed_data[0][0], a_address_str)
        self.assertEqual(parsed_data[1][0], a_address_str)

    def test_integration_formatted_address_to_pickle(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_parsed_address = FormattedParsedAddress({a_address_str: a_parsed_address})

        to_pickle(a_parsed_address, export_path=self.an_export_path)

        with open(self.an_export_path, "rb") as file:
            parsed_data = pickle.load(file)
        self.assertEqual(parsed_data[0][0], a_address_str)

    def test_integration_list_formatted_addresses_to_json(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_list_of_parsed_addresses = [
            FormattedParsedAddress({a_address_str: a_parsed_address}),
            FormattedParsedAddress({a_address_str: a_parsed_address}),
        ]

        to_json(a_list_of_parsed_addresses, export_path=self.an_export_path)

        with open(self.an_export_path, "r", encoding='utf-8') as file:
            parsed_data = json.load(file)
        self.assertIsInstance(parsed_data[0], dict)
        self.assertEqual(parsed_data[0].get("Address"), a_address_str)
        self.assertEqual(parsed_data[1].get("Address"), a_address_str)

    def test_integration_formatted_address_to_json(self):
        a_address_str = "3 test road"
        a_parsed_address = [
            ("3", "StreetNumber"),
            ("test", "StreetName"),
            ("road", "StreetName"),
        ]
        a_parsed_address = FormattedParsedAddress({a_address_str: a_parsed_address})

        to_json(a_parsed_address, export_path=self.an_export_path)

        with open(self.an_export_path, "r", encoding='utf-8') as file:
            parsed_data = json.load(file)
        self.assertIsInstance(parsed_data[0], dict)
        self.assertEqual(parsed_data[0].get("Address"), a_address_str)

    def test_generate_export_path_export_proper_path(self):
        a_export_file_name = "export.p"

        a_relative_dataset_path = os.path.join(".", "file_name.p")
        actual = generate_export_path(a_relative_dataset_path, a_export_file_name)
        expected = os.path.join(".", a_export_file_name)
        self.assertEqual(actual, expected)

        an_absolute_dataset_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        actual = generate_export_path(an_absolute_dataset_path, a_export_file_name)
        expected = os.path.join(self.temp_dir_obj.name, a_export_file_name)
        self.assertEqual(actual, expected)

    def test_replace_path_extension(self):
        a_relative_dataset_path = os.path.join(".", "file_name.p")
        actual = replace_path_extension(a_relative_dataset_path, ".log")
        expected = a_relative_dataset_path.replace(".p", ".log")

        self.assertEqual(actual, expected)

        an_absolute_dataset_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        actual = replace_path_extension(an_absolute_dataset_path, ".log")
        expected = an_absolute_dataset_path.replace(".p", ".log")

        self.assertEqual(actual, expected)

        an_absolute_dataset_path = os.path.join(self.temp_dir_obj.name, "an_export_file.p")
        actual = replace_path_extension(an_absolute_dataset_path, ".txt")
        expected = an_absolute_dataset_path.replace(".p", ".txt")

        self.assertEqual(actual, expected)

    def test_givenAnAttModel_whenHandlingAttentionModelType_returnTrue(self):
        a_att_parsing_model = "fasttext-attention"
        actual_update_args = attention_model_type_handling(a_att_parsing_model)
        self.assertTrue(actual_update_args.get("attention_mechanism"))

        a_att_parsing_model = "bpemb-attention"
        actual_update_args = attention_model_type_handling(a_att_parsing_model)
        self.assertTrue(actual_update_args.get("attention_mechanism"))

    def test_givenNotAnAttModel_whenHandlingAttentionModelType_returnFalse(self):
        not_a_att_parsing_model = "fasttext"
        actual_update_args = attention_model_type_handling(not_a_att_parsing_model)
        self.assertFalse(actual_update_args.get("attention_mechanism"))

        not_a_att_parsing_model = "bpemb"
        actual_update_args = attention_model_type_handling(not_a_att_parsing_model)
        self.assertFalse(actual_update_args.get("attention_mechanism"))

    def test_givenVariousTrueArgValue_whenCallBoolParse_thenReturnTrue(self):
        true_values_arg = ["true", "t", "yes", "y", "1"]

        for true_value in true_values_arg:
            self.assertTrue(bool_parse(arg=true_value))

    def test_givenVariousFalseArgValue_whenCallBoolParse_thenReturnFalse(self):
        false_values_arg = ["false", "f", "no", "n", "0"]

        for false_value in false_values_arg:
            self.assertFalse(bool_parse(arg=false_value))

    def test_givenNotVariousTrueOrFalseValue_whenCallBoolParse_thenRaiseError(self):
        wrong_values = ["tt", "nn", "10", "a_value", "bad", "value", "another"]

        for wrong_value in wrong_values:
            with self.assertRaises(argparse.ArgumentTypeError):
                bool_parse(wrong_value)

    @patch("deepparse.cli.tools.PickleDatasetContainer")
    def test_givenAParseSettingsPickle_whenDataContainerFactory_thenReturnDataContainer(self, dataset_container_mock):
        a_pickle_path = "a/pickle/path.p"

        data_container_factory(dataset_path=a_pickle_path, trainable_dataset=False)

        dataset_container_mock.assert_called()
        dataset_container_mock.assert_called_with(a_pickle_path, is_training_container=False)

    @patch("deepparse.cli.tools.CSVDatasetContainer")
    def test_givenAParseSettingsCSV_whenDataContainerFactory_thenReturnProperlySetDataContainer(
        self, dataset_container_mock
    ):
        a_csv_path = "a/pickle/path.csv"
        a_csv_column_name = "a_column_name"

        data_container_factory(dataset_path=a_csv_path, trainable_dataset=False, csv_column_name=a_csv_column_name)

        dataset_container_mock.assert_called()
        dataset_container_mock.assert_called_with(
            a_csv_path, column_names=a_csv_column_name, separator=None, is_training_container=False
        )

        a_separator = "\t"

        data_container_factory(
            dataset_path=a_csv_path,
            trainable_dataset=False,
            csv_column_name=a_csv_column_name,
            csv_column_separator=a_separator,
        )

        dataset_container_mock.assert_called()
        dataset_container_mock.assert_called_with(
            a_csv_path, column_names=a_csv_column_name, separator=a_separator, is_training_container=False
        )

    def test_givenAParseOrTrainWrongExtension_thenRaiseError(self):
        wrong_file_extension = "a/wrong/file/extension.txt"
        with self.assertRaises(ValueError):
            data_container_factory(dataset_path=wrong_file_extension, trainable_dataset=False)

        wrong_file_extension = "a/wrong/file/extension.txt"
        with self.assertRaises(ValueError):
            data_container_factory(dataset_path=wrong_file_extension, trainable_dataset=True)

    def test_givenAParseOrTrainCSVFileWithWrongSettings_thenRaiseError(self):
        a_csv_path = "a/pickle/path.csv"

        # No CSV columns names
        with self.assertRaises(ValueError):
            data_container_factory(dataset_path=a_csv_path, trainable_dataset=False)

        with self.assertRaises(ValueError):
            data_container_factory(dataset_path=a_csv_path, trainable_dataset=True)

    @patch("deepparse.cli.tools.PickleDatasetContainer")
    def test_givenATrainSettingsPickle_whenDataContainerFactory_thenReturnDataContainer(self, dataset_container_mock):
        a_pickle_path = "a/pickle/path.p"

        data_container_factory(dataset_path=a_pickle_path, trainable_dataset=True)

        dataset_container_mock.assert_called()
        dataset_container_mock.assert_called_with(a_pickle_path, is_training_container=True)

    @patch("deepparse.cli.tools.CSVDatasetContainer")
    def test_givenATrainSettingsCSV_whenDataContainerFactory_thenReturnProperlySetDataContainer(
        self, dataset_container_mock
    ):
        # Good for train, test and val type data
        a_csv_path = "a/pickle/path.csv"
        a_csv_column_names = ["a_column_name", "a_column_name"]

        data_container_factory(dataset_path=a_csv_path, trainable_dataset=True, csv_column_names=a_csv_column_names)

        dataset_container_mock.assert_called()
        dataset_container_mock.assert_called_with(
            a_csv_path, column_names=a_csv_column_names, separator=None, is_training_container=True
        )

        a_separator = "\t"

        data_container_factory(
            dataset_path=a_csv_path,
            trainable_dataset=True,
            csv_column_names=a_csv_column_names,
            csv_column_separator=a_separator,
        )

        dataset_container_mock.assert_called()
        dataset_container_mock.assert_called_with(
            a_csv_path, column_names=a_csv_column_names, separator=a_separator, is_training_container=True
        )
