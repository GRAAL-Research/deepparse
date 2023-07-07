# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-public-methods

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import gzip
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, mock_open, call, MagicMock
from unittest import TestCase
from urllib3.exceptions import MaxRetryError

import requests
from requests import HTTPError


from fasttext.FastText import _FastText

from deepparse.errors.server_error import ServerError
from deepparse.download_tools import (
    download_models,
    download_model,
    download_weights,
    download_fasttext_embeddings,
    download_fasttext_magnitude_embeddings,
    load_fasttext_embeddings,
    _print_progress,
    download_from_public_repository,
    latest_version,
)

from tests.base_file_exist import FileCreationTestCase
from tests.base_capture_output import CaptureOutputTestCase
from tests.tools import create_file


class FastTextToolsTests(CaptureOutputTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_directory_path = cls.temp_dir_obj.name
        cls.a_fasttext_file_name_path = os.path.join(cls.a_directory_path, "cc.fr.300.bin")
        cls.a_fasttext_gz_file_name_path = os.path.join(cls.a_directory_path, "cc.fr.300.bin.gz")
        cls.a_fasttext_light_name_path = os.path.join(cls.a_directory_path, "fasttext.magnitude")
        cls.a_fasttext_light_gz_file_name_path = os.path.join(cls.a_directory_path, "fasttext.magnitude.gz")

        # The payload is a first chunk "a", a second chunk "b" and a empty chunk ("") to end the loop
        cls.a_response_payload = ["a", "b", ""]

        cls.a_fake_embeddings_path = os.path.join(cls.temp_dir_obj.name, "fake_embeddings_cc.fr.300.bin")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def assertStdoutContains(self, values):
        for value in values:
            self.assertIn(value, self.test_out.getvalue().strip())

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsLocal_whenDownloadFasttextEmbeddings_thenReturnFilePath(self, isfile_mock):
        isfile_mock.return_value = True

        expected = self.a_fasttext_file_name_path
        actual = download_fasttext_embeddings(self.a_directory_path)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsLocal_whenDownloadFasttextEmbeddingsOffline_thenReturnFilePath(self, isfile_mock):
        isfile_mock.return_value = True

        expected = self.a_fasttext_file_name_path

        actual = download_fasttext_embeddings(self.a_directory_path, offline=True)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsLocal_whenDownloadFasttextEmbeddingsOffline_thenDontCallDownloadGZModel(
        self, isfile_mock
    ):
        isfile_mock.return_value = True

        with patch("deepparse.download_tools.download_gz_model") as download_gz_model_mock:
            download_fasttext_embeddings(self.a_directory_path, offline=True)
            download_gz_model_mock.assert_not_called()

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsOffline_thenReturnFilePath(
        self, isfile_mock
    ):
        isfile_mock.return_value = False

        expected = self.a_fasttext_file_name_path

        actual = download_fasttext_embeddings(self.a_directory_path, offline=True)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsOffline_thenDontCallDownloadGZModel(
        self, isfile_mock
    ):
        isfile_mock.return_value = False

        with patch("deepparse.download_tools.download_gz_model") as download_gz_model_mock:
            download_fasttext_embeddings(self.a_directory_path, offline=True)
            download_gz_model_mock.assert_not_called()

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsNotLocal_whenDownloadFasttextEmbeddings_thenDownloadIt(self, isfile_mock):
        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_file_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_file_name_path.encode("utf-8"))

        with patch("deepparse.download_tools.download_gz_model"):
            actual = download_fasttext_embeddings(self.a_directory_path)
            expected = self.a_fasttext_file_name_path
            self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsLocal_whenDownloadFasttextLightEmbeddings_thenReturnFilePath(
        self, isfile_mock
    ):
        isfile_mock.return_value = True
        expected = self.a_fasttext_light_name_path
        actual = download_fasttext_magnitude_embeddings(self.a_directory_path)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsLocal_whenDownloadFasttextEmbeddingsOffline_thenReturnFilePath(
        self, isfile_mock
    ):
        isfile_mock.return_value = True

        expected = self.a_fasttext_light_name_path

        actual = download_fasttext_magnitude_embeddings(self.a_directory_path, offline=True)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsLocal_whenDownloadFasttextEmbeddingsOffline_thenDontCallDownloadGZModel(
        self, isfile_mock
    ):
        isfile_mock.return_value = True

        with patch("deepparse.download_tools.download_from_public_repository") as download_model_mock:
            download_fasttext_magnitude_embeddings(self.a_directory_path, offline=True)
            download_model_mock.assert_not_called()

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsOffline_thenReturnFilePath(
        self, isfile_mock
    ):
        isfile_mock.return_value = False

        expected = self.a_fasttext_light_name_path

        actual = download_fasttext_magnitude_embeddings(self.a_directory_path, offline=True)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsOffline_thenDontCallDownloadGZModel(
        self, isfile_mock
    ):
        isfile_mock.return_value = False

        with patch("deepparse.download_tools.download_gz_model") as download_gz_model_mock:
            download_fasttext_magnitude_embeddings(self.a_directory_path, offline=True)
            download_gz_model_mock.assert_not_called()

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextLightEmbeddings_thenDownloadIt(
        self, isfile_mock
    ):
        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_light_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_light_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_light_name_path.encode("utf-8"))

        with patch("deepparse.download_tools.download_from_public_repository") as _:
            actual = download_fasttext_magnitude_embeddings(self.a_directory_path)
            expected = self.a_fasttext_light_name_path
            self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsNoVerbose_thenNoVerbose(
        self, isfile_mock
    ):
        self._capture_output()

        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_light_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_light_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_light_name_path.encode("utf-8"))

        with patch("deepparse.download_tools.download_from_public_repository"):
            download_fasttext_magnitude_embeddings(self.a_directory_path, verbose=False)

            expected = ""

            actual = self.test_out.getvalue().strip()
            self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsVerbose_thenVerbose(self, isfile_mock):
        self._capture_output()

        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_light_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_light_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_light_name_path.encode("utf-8"))

        with patch("deepparse.download_tools.download_from_public_repository"):
            download_fasttext_magnitude_embeddings(self.a_directory_path, verbose=True)

            expected = (
                "The fastText pretrained word embeddings will be download in magnitude format (2.3 GO), "
                "this process will take several minutes."
            )

            actual = self.test_out.getvalue().strip()
            self.assertEqual(expected, actual)

    def test_givenAFileToDownload_whenPrintProgress_thenPrintProperly(self):
        self._capture_output()

        a_total_size = 10
        for downloaded_bytes in range(1, a_total_size + 1):
            _print_progress(downloaded_bytes, a_total_size)

            # we verify some cases
            if downloaded_bytes == 1:
                self.assertEqual(
                    self.test_out.getvalue().strip(),
                    "(10.00%) [=====>                                             ]",
                )
            elif downloaded_bytes == 7:
                self.assertIn(
                    "(70.00%) [===================================>               ]",
                    self.test_out.getvalue().strip(),
                )
            elif downloaded_bytes == 10:
                self.assertIn(
                    "(100.00%) [==================================================>]",
                    self.test_out.getvalue().strip(),
                )

        self.assertStdoutContains(["[", ">", "=", "]"])

    @patch("builtins.open", new_callable=mock_open, read_data="a sentence")
    @patch("deepparse.download_tools.urlopen")
    @patch("deepparse.download_tools.gzip")
    @patch("deepparse.download_tools.os.rename")
    @patch("deepparse.download_tools.shutil")
    @patch("deepparse.download_tools.os.remove")
    def test_givenADownloadFasttext_whenPrintProgressSetToVerbose_thenDontPrint(
        self,
        os_remove_mock,
        shutil_mock,
        os_rename_mock,
        g_zip_mock,
        urlopen_mock,
        open_mock,
    ):
        # pylint: disable=too-many-arguments
        urlopen_mock().read.side_effect = self.a_response_payload
        self._capture_output()
        with urlopen_mock:
            _ = download_fasttext_embeddings(self.a_directory_path, verbose=False)

        expected = ""
        actual = self.test_out.getvalue().strip()

        self.assertEqual(expected, actual)

    @patch("builtins.open", new_callable=mock_open, read_data="a sentence")
    @patch("deepparse.download_tools.urlopen")
    @patch("deepparse.download_tools.gzip")
    @patch("deepparse.download_tools.os.rename")
    @patch("deepparse.download_tools.shutil")
    @patch("deepparse.download_tools.os.remove")
    def test_givenADownloadFasttext_whenPrintProgressSetToVerbose_thenPrint(
        self,
        os_remove_mock,
        shutil_mock,
        os_rename_mock,
        g_zip_mock,
        urlopen_mock,
        open_mock,
    ):
        # pylint: disable=too-many-arguments
        urlopen_mock().read.side_effect = self.a_response_payload
        urlopen_mock().getheader.return_value = "2"
        self._capture_output()
        with urlopen_mock:
            _ = download_fasttext_embeddings(self.a_directory_path, verbose=True)
        actual = self.test_out.getvalue().strip()

        expected = (
            "The fastText pretrained word embeddings will be downloaded (6.8 GO), "
            "this process will take several minutes."
        )
        self.assertIn(expected, actual)

        expected = "Downloading https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"
        self.assertIn(expected, actual)

        expected = "(50.00%) [=========================>                         ]"
        self.assertIn(expected, actual)

        expected = "(100.00%) [==================================================>]"
        self.assertIn(expected, actual)

    def test_givenAFasttextEmbeddingsToLoad_whenLoad_thenLoadProperly(self):
        download_from_public_repository("fake_embeddings_cc.fr.300", self.a_directory_path, "bin")
        embeddings_path = self.a_fake_embeddings_path

        embeddings = load_fasttext_embeddings(embeddings_path)

        self.assertIsInstance(embeddings, _FastText)


class DownloadModelsTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_dir = os.path.join(self.temp_dir_obj.name, "fake_cache")
        self.models_type_mapping = {
            "fasttext": "fasttext",
            "fasttext-attention": "fasttext_attention",
            "fasttext-light": "fasttext",
            "bpemb": "bpemb",
            "bpemb-attention": "bpemb_attention",
        }

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    @patch("deepparse.download_tools.download_weights")
    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.BPEmb")
    def test_givenADownloadAllModels_whenModelsAreNotLocal_thenDownloadAllModels(
        self,
        BPEmb_mock,
        download_fasttext_magnitude_embeddings_mock,
        download_fasttext_embeddings_mock,
        weights_download_mock,
    ):
        download_models(saving_cache_path=self.fake_cache_dir)

        download_fasttext_embeddings_mock.assert_called()
        download_fasttext_embeddings_mock.assert_called_with(cache_dir=self.fake_cache_dir)

        download_fasttext_magnitude_embeddings_mock.assert_called()
        download_fasttext_magnitude_embeddings_mock.assert_called_with(cache_dir=self.fake_cache_dir)

        BPEmb_mock.assert_called()
        BPEmb_mock.assert_called_with(lang="multi", vs=100000, dim=300, cache_dir=self.fake_cache_dir)

        weights_download_mock.assert_called()

        expected_call_count = 5
        actual_call_count = weights_download_mock.call_count
        self.assertEqual(expected_call_count, actual_call_count)

        for model_type in self.models_type_mapping.values():
            weights_download_mock.assert_has_calls([call(model_type, saving_dir=self.fake_cache_dir)])


class DownloadModelTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_dir = os.path.join(self.temp_dir_obj.name, "fake_cache")
        self.a_fasttext_model_type = "fasttext"
        self.a_fasttext_att_model_type = "fasttext-attention"
        self.a_fasttext_att_model_file_name = "fasttext_attention"
        self.a_fasttext_light_model_type = "fasttext-light"
        self.a_fasttext_light_model_file_name = "fasttext"
        self.a_bpemb_model_type = "bpemb"
        self.a_bpemb_att_model_type = "bpemb-attention"
        self.a_bpemb_att_model_type_file_name = "bpemb_attention"
        self.latest_fasttext_version = "617a417a2f2b02654f7deb5b5cbc60ab2f6334ba"
        self.latest_bpemb_version = "6d01367745157066ea6e621ac087be828137711f"

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    @patch("deepparse.download_tools.download_weights")
    def test_givenAFasttextDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_fasttext_embeddings") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_called()
                downloader.assert_called_with(cache_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_weights")
    def test_givenADownload_whenCachePathIsChange_thenDownloadInCacheDir(self, weights_download_mock):
        with patch("deepparse.download_tools.download_fasttext_embeddings") as downloader:
            download_model(self.a_fasttext_model_type, self.fake_cache_dir)

            downloader.assert_called()
            downloader.assert_called_with(cache_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_weights")
    def test_givenAFasttextMagnitudeDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_fasttext_magnitude_embeddings") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_called_with(cache_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_weights")
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.BPEmb") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(
                    lang="multi", vs=100000, dim=300, cache_dir=self.fake_cache_dir
                )  # settings for BPemb

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    def test_givenAFasttextDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    def test_givenAFasttextAttDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_att_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_att_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    def test_givenAFasttextLightDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_light_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    def test_givenABPembAttDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_att_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_att_model_type_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    @patch("deepparse.download_tools.latest_version", return_value=False)  # not the latest version
    def test_givenAFasttextDownload_whenModelIsLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock, latest_version_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    @patch("deepparse.download_tools.latest_version", return_value=False)  # not the latest version
    def test_givenAFasttextLightDownload_whenModelIsLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock, latest_version_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_light_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    @patch("deepparse.download_tools.latest_version", return_value=False)  # not the latest version
    def test_givenABPembDownload_whenModelIsLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock, latest_version_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    @patch("deepparse.download_tools.latest_version", return_value=True)  # the latest version
    def test_givenAFasttextDownload_whenModelIsLocalAndGoodVersion_thenDoNoting(
        self, download_embeddings_mock, os_is_file_mock, latest_version_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_not_called()

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    @patch("deepparse.download_tools.latest_version", return_value=True)  # the latest version
    def test_givenAFasttextLightDownload_whenModelIsLocalAndGoodVersion_thenDoNoting(
        self, download_embeddings_mock, os_is_file_mock, latest_version_mock
    ):
        os_is_file_mock.return_value = True
        latest_version_mock.return_value = True  # the latest version

        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_not_called()

    @patch("deepparse.download_tools.BPEmb")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    @patch("deepparse.download_tools.latest_version", return_value=True)  # the latest version
    def test_givenABPembDownload_whenModelIsLocalAndGoodVersion_thenDoNoting(
        self, download_embeddings_mock, os_is_file_mock, latest_version_mock
    ):
        os_is_file_mock.return_value = True
        latest_version_mock.return_value = True  # the latest version

        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_not_called()

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenAFasttextDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenAFasttextLightDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_light_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenABPembDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, saving_dir=self.fake_cache_dir)


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
        with patch("deepparse.download_tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = HTTPError(an_http_error_msg, response=response_mock)

            self.assertTrue(latest_version("bpemb", self.fake_cache_path, verbose=False))

    def test_givenANotHandledHTTPError_whenLatestVersionCall_thenRaiseError(self):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        an_http_error_msg = "An http error message"
        response_mock = MagicMock()
        response_mock.status_code = 300
        with patch("deepparse.download_tools.download_from_public_repository") as download_from_public_repository_mock:
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
        with patch("deepparse.download_tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = HTTPError(an_http_error_msg, response=response_mock)

            with self.assertWarns(RuntimeWarning):
                latest_version("bpemb", self.fake_cache_path, verbose=True)

    def test_givenANoInternetError_whenLatestVersionCall_thenReturnTrue(
        self,
    ):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        with patch("deepparse.download_tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = MaxRetryError(pool=MagicMock(), url=MagicMock())

            self.assertTrue(latest_version("bpemb", self.fake_cache_path, verbose=False))

    def test_givenANoInternetError_whenLatestVersionCall_thenPrintWarning(
        self,
    ):
        self.create_cache_version("bpemb", self.latest_fasttext_version)

        with patch("deepparse.download_tools.download_from_public_repository") as download_from_public_repository_mock:
            download_from_public_repository_mock.side_effect = MaxRetryError(pool=MagicMock(), url=MagicMock())

            with self.assertWarns(RuntimeWarning):
                latest_version("bpemb", self.fake_cache_path, verbose=True)

    @patch("deepparse.download_tools.os.path.exists", return_value=True)
    @patch("deepparse.download_tools.shutil.rmtree")
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
        with patch("deepparse.download_tools.download_from_public_repository") as downloader:
            download_weights(model_filename="fasttext", saving_dir="./", verbose=self.verbose)

            downloader.assert_any_call("fasttext", "./", "ckpt")
            downloader.assert_any_call("fasttext", "./", "version")

        with patch("deepparse.download_tools.download_from_public_repository") as downloader:
            download_weights(model_filename="bpemb", saving_dir="./", verbose=self.verbose)

            downloader.assert_any_call("bpemb", "./", "ckpt")
            downloader.assert_any_call("bpemb", "./", "version")

    def test_givenModelFasttextWeightsToDownloadVerbose_whenDownloadOk_thenVerbose(
        self,
    ):
        self._capture_output()
        with patch("deepparse.download_tools.download_from_public_repository"):
            download_weights(model_filename="fasttext", saving_dir="./", verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Downloading the pre-trained weights for the network fasttext."

        self.assertEqual(actual, expected)

    def test_givenModelBPEmbWeightsToDownloadVerbose_whenDownloadOk_thenVerbose(self):
        self._capture_output()
        with patch("deepparse.download_tools.download_from_public_repository"):
            download_weights(model_filename="bpemb", saving_dir="./", verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Downloading the pre-trained weights for the network bpemb."

        self.assertEqual(actual, expected)

    def test_givenModelWeightsToDownload_whenDownloadHTTPTimeOut_thenRaiseError(self):
        response_mock = MagicMock()

        with patch("deepparse.download_tools.download_from_public_repository") as downloader:
            downloader.side_effect = requests.exceptions.ConnectTimeout("An error message", response=response_mock)

            with self.assertRaises(ServerError):
                download_weights(model_filename="fasttext", saving_dir="./", verbose=self.verbose)


if __name__ == "__main__":
    unittest.main()
