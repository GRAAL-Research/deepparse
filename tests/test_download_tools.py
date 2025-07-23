# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-public-methods

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import gzip
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, mock_open, call
from unittest import TestCase

from gensim.models import FastText
from gensim.test.utils import common_texts
from gensim.models._fasttext_bin import save


from fasttext.FastText import _FastText

from deepparse.download_tools import (
    download_models,
    download_model,
    download_fasttext_embeddings,
    download_fasttext_magnitude_embeddings,
    load_fasttext_embeddings,
    _print_progress,
)

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

        model = FastText(vector_size=4, window=3, min_count=1)
        model.build_vocab(corpus_iterable=common_texts)
        model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=1)
        save(model, cls.a_fake_embeddings_path, {"lr_update_rate": 100, "word_ngrams": 1}, "utf-8")

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

    @patch("deepparse.download_tools.hf_hub_download")
    def test_givenAFasttextLightEmbeddings_whenDownloadFasttextLightEmbeddings_thenReturnFilePath(
        self, hub_download_mock
    ):
        hub_download_mock.return_value = self.a_fasttext_light_name_path

        expected = self.a_fasttext_light_name_path
        actual = download_fasttext_magnitude_embeddings(self.a_directory_path)
        self.assertEqual(expected, actual)

    @patch("deepparse.download_tools.hf_hub_download")
    def test_givenAFasttextLightEmbeddings_whenDownloadFasttextEmbeddingsOffline_thenReturnFilePath(
        self, hub_download_mock
    ):
        hub_download_mock.return_value = self.a_fasttext_light_name_path

        expected = self.a_fasttext_light_name_path

        actual = download_fasttext_magnitude_embeddings(self.a_directory_path, offline=True)
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
            "The FastText pretrained word embeddings will be downloaded (6.8 GO), "
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
        embeddings = load_fasttext_embeddings(self.a_fake_embeddings_path)

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
    @patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix")
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
            weights_download_mock.assert_has_calls([call(model_type, self.fake_cache_dir, verbose=True, offline=False)])


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
            with patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix") as downloader:
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
                downloader.assert_any_call(self.a_fasttext_model_type, self.fake_cache_dir, verbose=True, offline=False)

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    def test_givenAFasttextAttDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_att_model_type)

                downloader.assert_called()
                downloader.assert_any_call(
                    self.a_fasttext_att_model_file_name, self.fake_cache_dir, verbose=True, offline=False
                )

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    def test_givenAFasttextLightDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_any_call(
                    self.a_fasttext_light_model_file_name, self.fake_cache_dir, verbose=True, offline=False
                )

    @patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix")
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, self.fake_cache_dir, verbose=True, offline=False)

    @patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix")
    def test_givenABPembAttDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_att_model_type)

                downloader.assert_called()
                downloader.assert_any_call(
                    self.a_bpemb_att_model_type_file_name, self.fake_cache_dir, verbose=True, offline=False
                )

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    def test_givenAFasttextDownload_whenModelIsLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, self.fake_cache_dir, verbose=True, offline=False)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    def test_givenAFasttextLightDownload_whenModelIsLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_any_call(
                    self.a_fasttext_light_model_file_name, self.fake_cache_dir, verbose=True, offline=False
                )

    @patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix")
    @patch("deepparse.download_tools.os.path.isfile", return_value=True)
    def test_givenABPembDownload_whenModelIsLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, self.fake_cache_dir, verbose=True, offline=False)

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenAFasttextDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, self.fake_cache_dir, verbose=True, offline=False)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenAFasttextLightDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_fasttext_light_model_type)

                downloader.assert_called()
                downloader.assert_any_call(
                    self.a_fasttext_light_model_file_name, self.fake_cache_dir, verbose=True, offline=False
                )

    @patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenABPembDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model(self.a_bpemb_model_type)

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, self.fake_cache_dir, verbose=True, offline=False)


if __name__ == "__main__":
    unittest.main()
