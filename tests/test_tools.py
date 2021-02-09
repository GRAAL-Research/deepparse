# Since we use a patch to mock verify last we skip the unused argument error
# pylint: disable=W0613, too-many-public-methods

import os
import unittest
from unittest import TestCase
from unittest.mock import patch

import requests

from deepparse import download_from_url, latest_version, download_weights, indices_splitting, \
    handle_pre_trained_checkpoint, handle_poutyne_version, valid_poutyne_version
from deepparse import handle_checkpoint, CACHE_PATH
from tests.tools import create_file


class ToolsTests(TestCase):

    def setUp(self) -> None:
        self.fake_cache_path = "./"
        self.a_file_extension = "version"
        self.latest_fasttext_version = "b4f098bb8909b1c8a8d24eea07df3435"
        self.latest_bpemb_version = "ac0dc019748b6853dca412add7234203"
        self.a_seed = 42
        self.verbose = False

        self.a_model_type_checkpoint = "a_fake_model_type"
        self.a_fasttext_model_type_checkpoint = "fasttext"
        self.a_bpemb_model_type_checkpoint = "bpemb"

    def tearDown(self) -> None:
        if os.path.exists("fasttext.version"):
            os.remove("fasttext.version")
        if os.path.exists("bpemb.version"):
            os.remove("bpemb.version")

    def create_cache_version(self, model_name, content):
        version_file_path = os.path.join(self.fake_cache_path, model_name + ".version")
        create_file(version_file_path, content)

    def test_givenAFasttextLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version("fasttext", self.latest_fasttext_version)
        self.assertTrue(latest_version("fasttext", self.fake_cache_path))

    def test_givenAFasttextNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(self):
        self.create_cache_version("fasttext", "not_the_last_version")
        self.assertFalse(latest_version("fasttext", self.fake_cache_path))

    def test_givenABPEmbLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version("bpemb", self.latest_bpemb_version)
        self.assertTrue(latest_version("bpemb", self.fake_cache_path))

    def test_givenABPEmbNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(self):
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

    def test_givenABestCheckpoint_whenHandleCheckpoint_thenReturnBest(self):
        checkpoint = "best"

        actual = handle_checkpoint(checkpoint)
        expected = checkpoint

        self.assertEqual(actual, expected)

    def test_givenALastCheckpoint_whenHandleCheckpoint_thenReturnLast(self):
        checkpoint = "last"

        actual = handle_checkpoint(checkpoint)
        expected = checkpoint

        self.assertEqual(actual, expected)

    def test_givenAIntCheckpoint_whenHandleCheckpoint_thenReturn1(self):
        checkpoint = 1

        actual = handle_checkpoint(checkpoint)
        expected = checkpoint

        self.assertEqual(actual, expected)

    @patch("deepparse.tools.latest_version")
    def test_givenAFasttextCheckpoint_whenHandleCheckpoint_thenReturnCachedFasttextPath(self, latest_version_check):
        checkpoint = "fasttext"

        actual = handle_checkpoint(checkpoint)
        expected = os.path.join(CACHE_PATH, checkpoint + ".ckpt")

        self.assertEqual(actual, expected)

    @patch("deepparse.tools.latest_version")
    def test_givenABPEmbCheckpoint_whenHandleCheckpoint_thenReturnCachedBPEmbPath(self, latest_version_check):
        checkpoint = "bpemb"

        actual = handle_checkpoint(checkpoint)
        expected = os.path.join(CACHE_PATH, checkpoint + ".ckpt")

        self.assertEqual(actual, expected)

    def test_givenAStringCheckpoint_whenHandleCheckpoint_thenReturnSamePath(self):
        pickle_checkpoint = "/a/path/to/a/model.ckpt"

        actual = handle_checkpoint(pickle_checkpoint)
        expected = pickle_checkpoint

        self.assertEqual(actual, expected)

    def test_givenBadNamesCheckpoint_whenHandleCheckpoint_thenRaiseErrors(self):
        with self.assertRaises(ValueError):
            bad_best_checkpoint = "bests"
            handle_checkpoint(bad_best_checkpoint)

        with self.assertRaises(ValueError):
            bad_last_checkpoint = "lasts"
            handle_checkpoint(bad_last_checkpoint)

        with self.assertRaises(ValueError):
            string_int_bad_checkpoint = "1"
            handle_checkpoint(string_int_bad_checkpoint)

        with self.assertRaises(ValueError):
            bad_fasttext_checkpoint = "fasttexts"
            handle_checkpoint(bad_fasttext_checkpoint)

        with self.assertRaises(ValueError):
            bad_bpemb_checkpoint = "bpembds"
            handle_checkpoint(bad_bpemb_checkpoint)

        with self.assertRaises(ValueError):
            bad_pickle_extension_checkpoint = "/a/path/to/a/model.pck"
            handle_checkpoint(bad_pickle_extension_checkpoint)

    # test if splitting respect ratio splitting
    def test_givenADataset_whenIndicesSplittingRatio8020_thenSplitIndices80Train20Valid(self):
        number_of_data_points_in_dataset = 100
        train_ratio = 0.8
        expected_train_indices = [
            83, 53, 70, 45, 44, 39, 22, 80, 10, 0, 18, 30, 73, 33, 90, 4, 76, 77, 12, 31, 55, 88, 26, 42, 69, 15, 40,
            96, 9, 72, 11, 47, 85, 28, 93, 5, 66, 65, 35, 16, 49, 34, 7, 95, 27, 19, 81, 25, 62, 13, 24, 3, 17, 38, 8,
            78, 6, 64, 36, 89, 56, 99, 54, 43, 50, 67, 46, 68, 61, 97, 79, 41, 58, 48, 98, 57, 75, 32, 94, 59
        ]
        expected_valid_indices = [63, 84, 37, 29, 1, 52, 21, 2, 23, 87, 91, 74, 86, 82, 20, 60, 71, 14, 92, 51]
        expected_len_train_indices = 80
        expected_len_valid_indices = 20

        actual_train_indices, actual_valid_indices = indices_splitting(number_of_data_points_in_dataset,
                                                                       train_ratio,
                                                                       seed=self.a_seed)
        self.assertEqual(len(actual_train_indices), expected_len_train_indices)
        self.assertEqual(len(actual_valid_indices), expected_len_valid_indices)
        self.assertEqual(actual_train_indices, expected_train_indices)
        self.assertEqual(actual_valid_indices, expected_valid_indices)

    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionLowerThan12_givenHandlePreTrainedCheckpoint_thenRaiseError(self, poutyne_mock):
        poutyne_mock.version.__version__ = "1.1"

        with self.assertRaises(NotImplementedError):
            handle_pre_trained_checkpoint(self.a_model_type_checkpoint)

    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointFasttext_thenReturnFasttext(
            self, poutyne_mock, latest_version_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = handle_pre_trained_checkpoint(self.a_fasttext_model_type_checkpoint)
        expected = os.path.join(CACHE_PATH, f"{self.a_fasttext_model_type_checkpoint}.ckpt")
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointBPEmb_thenReturnBPEmb(
            self, poutyne_mock, latest_version_mock):
        poutyne_mock.version.__version__ = "1.2"

        actual = handle_pre_trained_checkpoint(self.a_bpemb_model_type_checkpoint)
        expected = os.path.join(CACHE_PATH, f"{self.a_bpemb_model_type_checkpoint}.ckpt")
        self.assertEqual(expected, actual)

    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointFasttextNotLatestVersion_thenRaiseWarning(
            self, poutyne_mock, latest_version_mock):
        latest_version_mock.return_value = False  # Not the latest version
        poutyne_mock.version.__version__ = "1.2"

        with self.assertWarns(UserWarning):
            handle_pre_trained_checkpoint(self.a_bpemb_model_type_checkpoint)

    @patch("deepparse.tools.latest_version")
    @patch("deepparse.tools.poutyne")
    def test_givenPoutyneVersionGreaterThan12_givenHandlePreTrainedCheckpointBPEmbNotLatestVersion_thenRaiseWarning(
            self, poutyne_mock, latest_version_mock):
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


if __name__ == "__main__":
    unittest.main()
