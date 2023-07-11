# Since we use a patch to mock verify last we skip the unused argument error
# pylint: disable=unused-argument, too-many-public-methods

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deepparse import (
    extract_package_version,
    valid_poutyne_version,
    validate_data_to_parse,
    DataError,
)
from tests.base_capture_output import CaptureOutputTestCase
from tests.base_file_exist import FileCreationTestCase


class ValidationsTests(CaptureOutputTestCase, FileCreationTestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_path = self.temp_dir_obj.name
        self.a_file_extension = "version"
        self.latest_fasttext_version = "f67a0517c70a314bdde0b8440f21139d"
        self.latest_bpemb_version = "aa32fa918494b461202157c57734c374"
        self.a_seed = 42
        self.verbose = False

        self.a_model_type_checkpoint = "a_fake_model_type"
        self.a_fasttext_model_type_checkpoint = "fasttext"
        self.a_bpemb_model_type_checkpoint = "bpemb"

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_1_1_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.1"

        actual = extract_package_version(package=poutyne_mock)
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_1_1_1_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.1.1"

        actual = extract_package_version(package=poutyne_mock)
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_1_dev_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = extract_package_version(package=poutyne_mock)
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_1_1_dev_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = extract_package_version(package=poutyne_mock)
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_2_givenHandlePoutyneVersion_thenReturnVersion1_2(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = extract_package_version(package=poutyne_mock)
        expected = "1.2"
        self.assertEqual(expected, actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_2_givenValidPoutyneVersion_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = valid_poutyne_version()
        self.assertTrue(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_2_dev_givenValidPoutyneVersion_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2.dev1+81b3c7b"

        actual = valid_poutyne_version()
        self.assertTrue(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_1_givenValidPoutyneVersion_thenReturnFalse(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1"

        actual = valid_poutyne_version()
        self.assertFalse(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_1_dev_givenValidPoutyneVersion_thenReturnFalse(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = valid_poutyne_version()
        self.assertFalse(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_8_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.8"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion1_11_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.11"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion2_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "2.0"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion2_11_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "2.11"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.validations.poutyne")
    def test_givenPoutyneVersion2_givenValidPoutyneVersion3_thenReturnFalse(self, poutyne_mock):
        poutyne_mock.version.__version__ = "2.0"

        actual = valid_poutyne_version(min_major=3, min_minor=0)
        self.assertFalse(actual)

    def test_integrationValidateDataToParse(self):
        valid_data = ["An address", "another address"]
        validate_data_to_parse(valid_data)

    def test_givenEmptyAddress_thenRaiseDataError(self):
        empty_data = ["An address", "", '']
        with self.assertRaises(DataError):
            validate_data_to_parse(empty_data)

    def test_givenWhiteSpaceAddress_thenRaiseDataError(self):
        whitespace_data = ["An address", " "]
        with self.assertRaises(DataError):
            validate_data_to_parse(whitespace_data)

        whitespace_data = ["An address", "   "]
        with self.assertRaises(DataError):
            validate_data_to_parse(whitespace_data)

    def test_givenNoneAddress_then_raiseDataError(self):
        none_data = ["An address", None]
        with self.assertRaises(DataError):
            validate_data_to_parse(none_data)

        none_data = [None]
        with self.assertRaises(DataError):
            validate_data_to_parse(none_data)

    def test_givenTupleAddressesToParse_then_raiseDataError(self):
        tuple_data = [("An address", 0), ("Another address", 1)]
        with self.assertRaises(DataError):
            validate_data_to_parse(tuple_data)


if __name__ == "__main__":
    unittest.main()
