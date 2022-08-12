# Since we use a patch to mock verify last we skip the unused argument error
# pylint: disable=unused-argument, too-many-public-methods

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

import requests
from requests import HTTPError
from urllib3.exceptions import MaxRetryError

from deepparse import (
    download_from_public_repository,
    latest_version,
    download_weights,
    handle_poutyne_version,
    valid_poutyne_version,
    validate_data_to_parse,
    DataError,
)
from tests.base_capture_output import CaptureOutputTestCase
from tests.base_file_exist import FileCreationTestCase
from tests.tools import create_file


class ToolsTests(CaptureOutputTestCase, FileCreationTestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_path = self.temp_dir_obj.name
        self.a_file_extension = "version"
        self.latest_fasttext_version = "b4f098bb8909b1c8a8d24eea07df3435"
        self.latest_bpemb_version = "ac0dc019748b6853dca412add7234203"
        self.a_seed = 42
        self.verbose = False

        self.a_model_type_checkpoint = "a_fake_model_type"
        self.a_fasttext_model_type_checkpoint = "fasttext"
        self.a_bpemb_model_type_checkpoint = "bpemb"

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def create_cache_version(self, model_name, content):
        version_file_path = os.path.join(self.fake_cache_path, model_name + ".version")
        create_file(version_file_path, content)

    def test_givenAFasttextLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version("fasttext", self.latest_fasttext_version)
        self.assertTrue(latest_version("fasttext", self.fake_cache_path, verbose=False))

    def test_givenAFasttextNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(
        self,
    ):
        self.create_cache_version("fasttext", "not_the_last_version")
        self.assertFalse(latest_version("fasttext", self.fake_cache_path, verbose=False))

    def test_givenABPEmbLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version("bpemb", self.latest_bpemb_version)
        self.assertTrue(latest_version("bpemb", self.fake_cache_path, verbose=False))

    def test_givenABPEmbNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(
        self,
    ):
        self.create_cache_version("bpemb", "not_the_last_version")
        self.assertFalse(latest_version("bpemb", self.fake_cache_path, verbose=False))

    def test_givenAHTTPError_whenLatestVersionCall_thenReturnTrue(
        self,
    ):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        an_http_error_msg = "An http error message"
        response_mock = MagicMock()
        response_mock.status_code = 400
        with patch("deepparse.tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = HTTPError(an_http_error_msg, response=response_mock)

            self.assertTrue(latest_version("bpemb", self.fake_cache_path, verbose=False))

    def test_givenANotHandledHTTPError_whenLatestVersionCall_thenRaiseError(self):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        an_http_error_msg = "An http error message"
        response_mock = MagicMock()
        response_mock.status_code = 300
        with patch("deepparse.tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = HTTPError(an_http_error_msg, response=response_mock)

            with self.assertRaises(HTTPError):
                latest_version("bpemb", self.fake_cache_path, verbose=False)

    def test_givenAHTTPErrorRemoteServer_whenLatestVersionCall_thenPrintWarning(
        self,
    ):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        an_http_error_msg = "An http error message"
        response_mock = MagicMock()
        response_mock.status_code = 400
        with patch("deepparse.tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = HTTPError(an_http_error_msg, response=response_mock)

            with self.assertWarns(UserWarning):
                latest_version("bpemb", self.fake_cache_path, verbose=True)

    def test_givenANoInternetError_whenLatestVersionCall_thenReturnTrue(
        self,
    ):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        with patch("deepparse.tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = MaxRetryError(pool=MagicMock(), url=MagicMock())

            self.assertTrue(latest_version("bpemb", self.fake_cache_path, verbose=False))

    def test_givenANoInternetError_whenLatestVersionCall_thenPrintWarning(
        self,
    ):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        with patch("deepparse.tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = MaxRetryError(pool=MagicMock(), url=MagicMock())

            with self.assertWarns(UserWarning):
                latest_version("bpemb", self.fake_cache_path, verbose=True)

    @patch("deepparse.tools.os.path.exists", return_value=True)
    @patch("deepparse.tools.shutil.rmtree")
    def test_givenAModelVersion_whenVerifyIfLastVersion_thenCleanTmpRepo(self, os_path_exists_mock, shutil_rmtree_mock):
        self.create_cache_version("bpemb", "a_version")
        latest_version("bpemb", self.fake_cache_path, verbose=False)

        os_path_exists_mock.assert_called()
        shutil_rmtree_mock.assert_called()

    def test_givenFasttextVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "fasttext"

        download_from_public_repository(file_name, self.fake_cache_path, self.a_file_extension)

        self.assertFileExist(os.path.join(self.fake_cache_path, f"{file_name}.{self.a_file_extension}"))

    def test_givenFasttextVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_fasttext"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_public_repository(wrong_file_name, self.fake_cache_path, self.a_file_extension)

    def test_givenBPEmbVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "bpemb"

        download_from_public_repository(file_name, self.fake_cache_path, self.a_file_extension)

        self.assertFileExist(os.path.join(self.fake_cache_path, f"{file_name}.{self.a_file_extension}"))

    def test_givenBPEmbVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_bpemb"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_public_repository(wrong_file_name, self.fake_cache_path, self.a_file_extension)

    def test_givenModelWeightsToDownload_whenDownloadOk_thenWeightsAreDownloaded(self):
        with patch("deepparse.tools.download_from_public_repository") as downloader:
            download_weights(model="fasttext", saving_dir="./", verbose=self.verbose)

            downloader.assert_any_call("fasttext", "./", "ckpt")
            downloader.assert_any_call("fasttext", "./", "version")

        with patch("deepparse.tools.download_from_public_repository") as downloader:
            download_weights(model="bpemb", saving_dir="./", verbose=self.verbose)

            downloader.assert_any_call("bpemb", "./", "ckpt")
            downloader.assert_any_call("bpemb", "./", "version")

    def test_givenModelFasttextWeightsToDownloadVerbose_whenDownloadOk_thenVerbose(
        self,
    ):
        self._capture_output()
        with patch("deepparse.tools.download_from_public_repository"):
            download_weights(model="fasttext", saving_dir="./", verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Downloading the weights for the network fasttext."

        self.assertEqual(actual, expected)

    def test_givenModelBPEmbWeightsToDownloadVerbose_whenDownloadOk_thenVerbose(self):
        self._capture_output()
        with patch("deepparse.tools.download_from_public_repository"):
            download_weights(model="bpemb", saving_dir="./", verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Downloading the weights for the network bpemb."

        self.assertEqual(actual, expected)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_1_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.1"

        actual = handle_poutyne_version()
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_1_1_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.1.1"

        actual = handle_poutyne_version()
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_dev_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = handle_poutyne_version()
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_1_dev_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = handle_poutyne_version()
        expected = "1.1"
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_2_givenHandlePoutyneVersion_thenReturnVersion1_2(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = handle_poutyne_version()
        expected = "1.2"
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_2_givenValidPoutyneVersion_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = valid_poutyne_version()
        self.assertTrue(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_2_dev_givenValidPoutyneVersion_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2.dev1+81b3c7b"

        actual = valid_poutyne_version()
        self.assertTrue(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_givenValidPoutyneVersion_thenReturnFalse(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1"

        actual = valid_poutyne_version()
        self.assertFalse(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_dev_givenValidPoutyneVersion_thenReturnFalse(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = valid_poutyne_version()
        self.assertFalse(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_8_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.8"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_11_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.11"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion2_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "2.0"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion2_11_givenValidPoutyneVersion1_8_thenReturnTrue(self, poutyne_mock):
        poutyne_mock.version.__version__ = "2.11"

        actual = valid_poutyne_version(min_major=1, min_minor=8)
        self.assertTrue(actual)

    @patch("deepparse.tools.poutyne")
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
