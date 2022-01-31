# Since we use a patch to mock verify last we skip the unused argument error
# pylint: disable=unused-argument, too-many-public-methods

import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests

from deepparse import (
    download_from_url,
    latest_version,
    download_weights,
    handle_pre_trained_checkpoint,
    handle_poutyne_version,
    valid_poutyne_version,
    validate_data_to_parse,
    DataError,
)
from deepparse import handle_model_path, CACHE_PATH
from tests.base_capture_output import CaptureOutputTestCase
from tests.tools import create_file


class ToolsTests(CaptureOutputTestCase):
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
        self.assertTrue(latest_version("fasttext", self.fake_cache_path))

    def test_givenAFasttextNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(
        self,
    ):
        self.create_cache_version("fasttext", "not_the_last_version")
        self.assertFalse(latest_version("fasttext", self.fake_cache_path))

    def test_givenABPEmbLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version("bpemb", self.latest_bpemb_version)
        self.assertTrue(latest_version("bpemb", self.fake_cache_path))

    def test_givenABPEmbNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(
        self,
    ):
        self.create_cache_version("bpemb", "not_the_last_version")
        self.assertFalse(latest_version("bpemb", self.fake_cache_path))

    def test_givenFasttextVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "fasttext"

        download_from_url(file_name, self.fake_cache_path, self.a_file_extension)

        self.assertTrue(os.path.exists(os.path.join(self.fake_cache_path, f"{file_name}.{self.a_file_extension}")))

    def test_givenFasttextVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_fasttext"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_url(wrong_file_name, self.fake_cache_path, self.a_file_extension)

    def test_givenBPEmbVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "bpemb"

        download_from_url(file_name, self.fake_cache_path, self.a_file_extension)

        self.assertTrue(os.path.exists(os.path.join(self.fake_cache_path, f"{file_name}.{self.a_file_extension}")))

    def test_givenBPEmbVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_bpemb"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_url(wrong_file_name, self.fake_cache_path, self.a_file_extension)

    def test_givenModelWeightsToDownload_whenDownloadOk_thenWeightsAreDownloaded(self):
        with patch("deepparse.tools.download_from_url") as downloader:
            download_weights(model="fasttext", saving_dir="./", verbose=self.verbose)

            downloader.assert_any_call("fasttext", "./", "ckpt")
            downloader.assert_any_call("fasttext", "./", "version")

        with patch("deepparse.tools.download_from_url") as downloader:
            download_weights(model="bpemb", saving_dir="./", verbose=self.verbose)

            downloader.assert_any_call("bpemb", "./", "ckpt")
            downloader.assert_any_call("bpemb", "./", "version")

    def test_givenModelFasttextWeightsToDownloadVerbose_whenDownloadOk_thenVerbose(
        self,
    ):
        self._capture_output()
        with patch("deepparse.tools.download_from_url"):
            download_weights(model="fasttext", saving_dir="./", verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Downloading the weights for the network fasttext."

        self.assertEqual(actual, expected)

    def test_givenModelBPEmbWeightsToDownloadVerbose_whenDownloadOk_thenVerbose(self):
        self._capture_output()
        with patch("deepparse.tools.download_from_url"):
            download_weights(model="bpemb", saving_dir="./", verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Downloading the weights for the network bpemb."

        self.assertEqual(actual, expected)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    def test_givenAFasttextCheckpoint_whenHandleCheckpoint_thenReturnCachedFasttextPath(
        self, latest_version_check, isfile_mock
    ):
        isfile_mock.return_value = True
        checkpoint = "fasttext"

        actual = handle_model_path(checkpoint)
        expected = os.path.join(CACHE_PATH, checkpoint + ".ckpt")

        self.assertEqual(actual, expected)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    def test_givenABPEmbCheckpoint_whenHandleCheckpoint_thenReturnCachedBPEmbPath(
        self, latest_version_check, isfile_mock
    ):
        isfile_mock.return_value = True
        checkpoint = "bpemb"

        actual = handle_model_path(checkpoint)
        expected = os.path.join(CACHE_PATH, checkpoint + ".ckpt")

        self.assertEqual(actual, expected)

    def test_givenAStringCheckpoint_whenHandleCheckpoint_thenReturnSamePath(self):
        pickle_checkpoint = "/a/path/to/a/model.ckpt"

        actual = handle_model_path(pickle_checkpoint)
        expected = pickle_checkpoint

        self.assertEqual(actual, expected)

    def test_givenBadNamesCheckpoint_whenHandleCheckpoint_thenRaiseErrors(self):
        with self.assertRaises(ValueError):
            bad_best_checkpoint = "bests"
            handle_model_path(bad_best_checkpoint)

        with self.assertRaises(ValueError):
            bad_last_checkpoint = "lasts"
            handle_model_path(bad_last_checkpoint)

        with self.assertRaises(ValueError):
            string_int_bad_checkpoint = "1"
            handle_model_path(string_int_bad_checkpoint)

        with self.assertRaises(ValueError):
            bad_fasttext_checkpoint = "fasttexts"
            handle_model_path(bad_fasttext_checkpoint)

        with self.assertRaises(ValueError):
            bad_bpemb_checkpoint = "bpembds"
            handle_model_path(bad_bpemb_checkpoint)

        with self.assertRaises(ValueError):
            bad_pickle_extension_checkpoint = "/a/path/to/a/model.pck"
            handle_model_path(bad_pickle_extension_checkpoint)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionLowerThan12_givenHandlePreTrainedCheckpoint_thenRaiseError(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1"

        with self.assertRaises(NotImplementedError):
            handle_pre_trained_checkpoint(self.a_model_type_checkpoint)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointFasttext_thenReturnFasttext(
        self, poutyne_mock, latest_version_mock, isfile_mock
    ):
        poutyne_mock.version.__version__ = "1.2"
        isfile_mock.return_value = True

        actual = handle_pre_trained_checkpoint(self.a_fasttext_model_type_checkpoint)
        expected = os.path.join(CACHE_PATH, f"{self.a_fasttext_model_type_checkpoint}.ckpt")
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointFasttextNoLocalFile_thenReturnFasttext(
        self, poutyne_mock, latest_version_mock, isfile_mock
    ):
        poutyne_mock.version.__version__ = "1.2"
        isfile_mock.return_value = False

        with patch("deepparse.tools.download_weights"):
            actual = handle_pre_trained_checkpoint(self.a_fasttext_model_type_checkpoint)
        expected = os.path.join(CACHE_PATH, f"{self.a_fasttext_model_type_checkpoint}.ckpt")
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointBPEmb_thenReturnBPEmb(
        self, poutyne_mock, latest_version_mock, isfile_mock
    ):
        poutyne_mock.version.__version__ = "1.2"
        isfile_mock.return_value = True

        actual = handle_pre_trained_checkpoint(self.a_bpemb_model_type_checkpoint)
        expected = os.path.join(CACHE_PATH, f"{self.a_bpemb_model_type_checkpoint}.ckpt")
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointBPEmbNoLocalFile_thenReturnBPEmb(
        self, poutyne_mock, latest_version_mock, isfile_mock
    ):
        poutyne_mock.version.__version__ = "1.2"
        isfile_mock.return_value = False

        with patch("deepparse.tools.download_weights"):
            actual = handle_pre_trained_checkpoint(self.a_fasttext_model_type_checkpoint)
        expected = os.path.join(CACHE_PATH, f"{self.a_fasttext_model_type_checkpoint}.ckpt")
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointFasttextNotLatestVersion_thenRaiseWarning(
        self, poutyne_mock, latest_version_mock, isfile_mock
    ):
        isfile_mock.return_value = True
        latest_version_mock.return_value = False  # Not the latest version
        poutyne_mock.version.__version__ = "1.2"

        with self.assertWarns(UserWarning):
            handle_pre_trained_checkpoint(self.a_bpemb_model_type_checkpoint)

    @patch("os.path.isfile")
    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointBPEmbNotLatestVersion_thenRaiseWarning(
        self, poutyne_mock, latest_version_mock, isfile_mock
    ):
        isfile_mock.return_value = True
        latest_version_mock.return_value = False  # Not the latest version
        poutyne_mock.version.__version__ = "1.2"

        with self.assertWarns(UserWarning):
            handle_pre_trained_checkpoint(self.a_bpemb_model_type_checkpoint)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_1_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.1"

        actual = handle_poutyne_version()
        expected = 1.1
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_1_1_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.1.1"

        actual = handle_poutyne_version()
        expected = 1.1
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_dev_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = handle_poutyne_version()
        expected = 1.1
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_1_1_dev_givenHandlePoutyneVersion_thenReturnVersion1_1(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1.dev1+81b3c7b"

        actual = handle_poutyne_version()
        expected = 1.1
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersion1_2_givenHandlePoutyneVersion_thenReturnVersion1_2(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = handle_poutyne_version()
        expected = 1.2
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


if __name__ == "__main__":
    unittest.main()
