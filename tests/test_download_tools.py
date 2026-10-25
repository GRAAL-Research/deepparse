# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-public-methods

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import gzip
import os
import unittest
import warnings
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, mock_open, patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    pa = None
    pq = None
    PYARROW_AVAILABLE = False

from deepparse.dataset_container import ListDatasetContainer
from deepparse.embeddings_models import FastTextBinReader
from deepparse.download_tools import (
    _print_progress,
    download_fasttext_embeddings,
    download_fasttext_magnitude_embeddings,
    download_model,
    download_model2vec_embeddings,
    download_models,
    download_worldwide_addresses_sample,
    get_model2vec_model_type,
    load_fasttext_embeddings,
)
from tests.base_capture_output import CaptureOutputTestCase
from tests.fixtures.fasttext_bin_writer import create_tiny_fasttext_bin
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

        create_tiny_fasttext_bin(cls.a_fake_embeddings_path)

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
        with self.assertLogs("deepparse.download_tools", level="INFO") as cm:
            with urlopen_mock:
                _ = download_fasttext_embeddings(self.a_directory_path, verbose=True)

        log_messages = " ".join(cm.output)

        expected = (
            "The FastText pretrained word embeddings will be downloaded (6.8 GO), "
            "this process will take several minutes."
        )
        self.assertIn(expected, log_messages)

        expected = "Downloading https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"
        self.assertIn(expected, log_messages)

        # Progress bar still uses sys.stdout.write
        actual = self.test_out.getvalue().strip()
        expected = "(50.00%) [=========================>                         ]"
        self.assertIn(expected, actual)

        expected = "(100.00%) [==================================================>]"
        self.assertIn(expected, actual)

    def test_givenAFasttextEmbeddingsToLoad_whenLoad_thenLoadWithTheBinReaderAndWarn(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            embeddings = load_fasttext_embeddings(self.a_fake_embeddings_path)

        self.assertIsInstance(embeddings, FastTextBinReader)
        self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in caught_warnings))

    def test_whenImportFasttextAvailable_thenStillExposedForBackwardCompatibility(self):
        from deepparse import FASTTEXT_AVAILABLE  # pylint: disable=import-outside-toplevel

        self.assertIsInstance(FASTTEXT_AVAILABLE, bool)


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
            "model2vec-8m": "model2vec-8m",
            "model2vec-8m-attention": "model2vec-8m_attention",
            "model2vec-32m": "model2vec-32m",
            "model2vec-32m-attention": "model2vec-32m_attention",
            "model2vec-multilingual": "model2vec-multilingual",
            "model2vec-multilingual-attention": "model2vec-multilingual_attention",
        }

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    @patch("deepparse.download_tools.download_weights")
    @patch("deepparse.download_tools.download_model2vec_embeddings")
    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.BPEmbBaseURLWrapperBugFix")
    def test_givenADownloadAllModels_whenModelsAreNotLocal_thenDownloadAllModels(
        self,
        BPEmb_mock,
        download_fasttext_magnitude_embeddings_mock,
        download_fasttext_embeddings_mock,
        download_model2vec_embeddings_mock,
        weights_download_mock,
    ):
        download_models(saving_cache_path=self.fake_cache_dir)

        download_fasttext_embeddings_mock.assert_called()
        download_fasttext_embeddings_mock.assert_called_with(cache_dir=self.fake_cache_dir)

        download_fasttext_magnitude_embeddings_mock.assert_called()
        download_fasttext_magnitude_embeddings_mock.assert_called_with(cache_dir=self.fake_cache_dir)

        BPEmb_mock.assert_called()
        BPEmb_mock.assert_called_with(lang="multi", vs=100000, dim=300, cache_dir=self.fake_cache_dir)

        self.assertEqual(6, download_model2vec_embeddings_mock.call_count)

        weights_download_mock.assert_called()

        expected_call_count = len(self.models_type_mapping)
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


@unittest.skipUnless(PYARROW_AVAILABLE, "The pyarrow package is needed to build the parquet fixtures.")
class DownloadWorldwideAddressesSampleTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_country_code = "ca"
        cls.a_dataset_dir = f"datasets/deepparse/worldwide-addresses/{cls.a_country_code}"

        # The first chunks only hold postal codes, as the beginning of the real dataset.
        a_postal_code_chunk_rows = [("G1V 0A6 Canada", ["PostalCode", "PostalCode", "Country"])]
        # The last chunk (chunk-10 and not chunk-2, since the order is numeric) holds the street addresses.
        a_last_chunk_rows = [
            ("350 rue des Lilas Quebec", ["StreetNumber", "StreetName", "StreetName", "StreetName", "Municipality"]),
            ("Canada", ["Country"]),
            ("12 Main St Vanier Ottawa", ["StreetNumber", "StreetName", "StreetName", "Suburb", "Municipality"]),
            (
                "1 King Rd Kingston Frontenac County Canada",
                ["StreetNumber", "StreetName", "StreetName", "Municipality", "County", "County", "Country"],
            ),
            (
                "5 Elm St Downtown Halifax NS",
                ["StreetNumber", "StreetName", "StreetName", "District", "Municipality", "Province"],
            ),
        ]
        cls.chunk_files = {
            "chunk-0.parquet": a_postal_code_chunk_rows,
            "chunk-10.parquet": a_last_chunk_rows,
            "chunk-2.parquet": a_postal_code_chunk_rows,
        }
        for file_name, rows in cls.chunk_files.items():
            table = pa.table(
                {
                    "Address": [address for address, _ in rows],
                    "Tags": [tags for _, tags in rows],
                    "Language": ["eng"] * len(rows),
                }
            )
            # A row group size of 2 gives many row groups to test the reading from the end of the file.
            pq.write_table(table, os.path.join(cls.temp_dir_obj.name, file_name), row_group_size=2)

        # The "Canada" sample is dropped since it is empty once the Country words are removed.
        cls.expected_samples = [
            ("350 rue des Lilas Quebec", ["StreetNumber", "StreetName", "StreetName", "StreetName", "Municipality"]),
            ("12 Main St Vanier Ottawa", ["StreetNumber", "StreetName", "StreetName", "Municipality", "Municipality"]),
            ("1 King Rd Kingston", ["StreetNumber", "StreetName", "StreetName", "Municipality"]),
            (
                "5 Elm St Downtown Halifax NS",
                ["StreetNumber", "StreetName", "StreetName", "Municipality", "Municipality", "Province"],
            ),
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def _open_local_chunk(self, path, mode):
        return open(os.path.join(self.temp_dir_obj.name, os.path.basename(path)), mode)

    def _mock_file_system(self, hf_file_system_mock):
        file_system = hf_file_system_mock.return_value
        file_system.glob.return_value = [f"{self.a_dataset_dir}/{file_name}" for file_name in self.chunk_files]
        file_system.open.side_effect = self._open_local_chunk
        return file_system

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenACountryCode_whenDownloadSample_thenReadTheLastChunkOfTheCountry(self, hf_file_system_mock):
        file_system = self._mock_file_system(hf_file_system_mock)

        download_worldwide_addresses_sample(self.a_country_code.upper(), num_samples=2)

        file_system.glob.assert_called_once_with(f"{self.a_dataset_dir}/chunk-*.parquet")
        file_system.open.assert_called_once_with(f"{self.a_dataset_dir}/chunk-10.parquet", "rb")

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenEnoughData_whenDownloadSample_thenReturnTheLastMappedSamples(self, hf_file_system_mock):
        self._mock_file_system(hf_file_system_mock)

        actual = download_worldwide_addresses_sample(self.a_country_code, num_samples=3)

        self.assertEqual(self.expected_samples[-3:], actual)

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenNotEnoughData_whenDownloadSample_thenReturnAllTheMappedSamples(self, hf_file_system_mock):
        self._mock_file_system(hf_file_system_mock)

        actual = download_worldwide_addresses_sample(self.a_country_code, num_samples=100)

        self.assertEqual(self.expected_samples, actual)

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenSamples_whenDownloadSample_thenOnlyDeepparseTagsAreReturned(self, hf_file_system_mock):
        self._mock_file_system(hf_file_system_mock)

        actual = download_worldwide_addresses_sample(self.a_country_code, num_samples=100)

        deepparse_tags = {"StreetNumber", "StreetName", "Unit", "Municipality", "Province", "PostalCode", "Orientation"}
        for address, tags in actual:
            self.assertEqual(len(address.split()), len(tags))
            self.assertTrue(set(tags).issubset(deepparse_tags))

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenSamples_whenDownloadSample_thenUsableByAListDatasetContainer(self, hf_file_system_mock):
        self._mock_file_system(hf_file_system_mock)

        container = ListDatasetContainer(download_worldwide_addresses_sample(self.a_country_code, num_samples=100))

        self.assertEqual(len(self.expected_samples), len(container))

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenAnUnknownCountryCode_whenDownloadSample_thenRaiseValueError(self, hf_file_system_mock):
        hf_file_system_mock.return_value.glob.return_value = []

        with self.assertRaises(ValueError):
            download_worldwide_addresses_sample("zz")

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenANonPositiveNumberOfSamples_whenDownloadSample_thenRaiseValueError(self, hf_file_system_mock):
        with self.assertRaises(ValueError):
            download_worldwide_addresses_sample(self.a_country_code, num_samples=0)

        hf_file_system_mock.assert_not_called()

    @patch("deepparse.download_tools.HfFileSystem")
    def test_givenPyarrowNotInstalled_whenDownloadSample_thenRaiseImportError(self, hf_file_system_mock):
        with patch("deepparse.download_tools.PYARROW_AVAILABLE", False):
            with self.assertRaises(ImportError):
                download_worldwide_addresses_sample(self.a_country_code)

        hf_file_system_mock.assert_not_called()


class Model2VecDownloadToolsTests(TestCase):
    def test_givenModel2VecModelTypes_whenGettingTheBaseModelType_thenRemoveTheAttentionSuffix(self):
        self.assertEqual("model2vec-8m", get_model2vec_model_type("model2vec-8m"))
        self.assertEqual("model2vec-32m", get_model2vec_model_type("model2vec-32m_attention"))
        self.assertEqual("model2vec-multilingual", get_model2vec_model_type("model2vec-multilingual-attention"))

    def test_givenANonModel2VecModelType_whenGettingTheBaseModelType_thenRaiseError(self):
        with self.assertRaises(ValueError):
            get_model2vec_model_type("fasttext")

    @patch("deepparse.download_tools.snapshot_download", return_value="a/local/path")
    def test_givenAModel2VecModelType_whenDownloading_thenOnlyDownloadTheFilesModel2VecLoads(self, snapshot_mock):
        path, dim = download_model2vec_embeddings("model2vec-32m_attention", cache_dir="a_cache", verbose=False)

        self.assertEqual(("a/local/path", 512), (path, dim))
        snapshot_mock.assert_called_once_with(
            "minishlab/potion-base-32M",
            cache_dir="a_cache",
            local_files_only=False,
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json"],
        )

    @patch("deepparse.download_tools.download_weights")
    @patch("deepparse.download_tools.download_model2vec_embeddings")
    def test_givenAModel2VecDownload_whenDownloadingTheModel_thenDownloadTheModel2VecEmbeddings(
        self, embeddings_download_mock, weights_download_mock
    ):
        download_model("model2vec-8m-attention", "a_cache")

        embeddings_download_mock.assert_called_once_with("model2vec-8m-attention", cache_dir="a_cache")
        weights_download_mock.assert_called_once_with("model2vec-8m_attention", "a_cache", verbose=True, offline=False)


if __name__ == "__main__":
    unittest.main()
