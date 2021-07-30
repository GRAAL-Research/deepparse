# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

import gzip
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, mock_open

from fasttext.FastText import _FastText

from deepparse import download_fasttext_embeddings, download_fasttext_magnitude_embeddings, download_from_url, \
    load_fasttext_embeddings
from deepparse.fasttext_tools import _print_progress
from tests.base_capture_output import CaptureOutputTestCase
from tests.tools import create_file


class ToolsTests(CaptureOutputTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_directory_path = os.path.join(cls.temp_dir_obj.name, "./")
        cls.a_fasttext_file_name_path = os.path.join(cls.a_directory_path, "cc.fr.300.bin")
        cls.a_fasttext_gz_file_name_path = os.path.join(cls.a_directory_path, "cc.fr.300.bin.gz")
        cls.a_fasttext_light_name_path = os.path.join(cls.a_directory_path, "fasttext.magnitude")
        cls.a_fasttext_light_gz_file_name_path = os.path.join(cls.a_directory_path, "fasttext.magnitude.gz")

        # the payload is a first "chunk" a, a second chunk "b" and a empty chunk "" to end the loop
        cls.a_response_payload = ["a", "b", ""]

        cls.a_fake_embeddings_path = os.path.join(cls.temp_dir_obj.name, "fake_embeddings_cc.fr.300.bin")

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def assertStdoutContains(self, values):
        for value in values:
            self.assertIn(value, self.test_out.getvalue().strip())

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsLocal_whenDownloadFasttextEmbeddings_thenReturnFilePath(self, isfile_mock):
        expected = self.a_fasttext_file_name_path
        actual = download_fasttext_embeddings(self.a_directory_path)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextEmbeddingsNotLocal_whenDownloadFasttextEmbeddings_thenDownloadIt(self, isfile_mock):
        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_file_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_file_name_path.encode("utf-8"))

        with patch("deepparse.fasttext_tools.download_gz_model"):
            actual = download_fasttext_embeddings(self.a_directory_path)
            expected = self.a_fasttext_file_name_path
            self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsLocal_whenDownloadFasttextLightEmbeddings_thenReturnFilePath(
            self, isfile_mock):
        expected = self.a_fasttext_light_name_path
        actual = download_fasttext_magnitude_embeddings(self.a_directory_path)
        self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextLightEmbeddings_thenDownloadIt(
            self, isfile_mock):
        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_light_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_light_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_light_name_path.encode("utf-8"))

        with patch("deepparse.fasttext_tools.download_from_url") as _:
            actual = download_fasttext_magnitude_embeddings(self.a_directory_path)
            expected = self.a_fasttext_light_name_path
            self.assertEqual(expected, actual)

    @patch("os.path.isfile")
    def test_givenAFasttextLightEmbeddingsNotLocal_whenDownloadFasttextEmbeddingsNoVerbose_thenNoVerbose(
            self, isfile_mock):
        self._capture_output()

        # since we create a local fake file, the file exist, so we mock that the file doest not exist.
        isfile_mock.return_value = False
        create_file(self.a_fasttext_light_name_path, content="Fake fasttext embedding content")

        # we create a fake fasttext archive
        with gzip.open(self.a_fasttext_light_gz_file_name_path, "wb") as f:
            f.write(self.a_fasttext_light_name_path.encode("utf-8"))

        with patch("deepparse.fasttext_tools.download_from_url"):
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

        with patch("deepparse.fasttext_tools.download_from_url"):
            download_fasttext_magnitude_embeddings(self.a_directory_path, verbose=True)

            expected = "The fastText pre-trained word embeddings will be download in magnitude format (2.3 GO), " \
                       "this process will take several minutes."

            actual = self.test_out.getvalue().strip()
            self.assertEqual(expected, actual)

    def test_givenAFileToDownload_whenPrintProgress_thenPrintProperly(self):
        self._capture_output()

        a_total_size = 10
        for downloaded_bytes in range(1, a_total_size + 1):
            _print_progress(downloaded_bytes, a_total_size)

            # we verify some cases
            if downloaded_bytes == 1:
                self.assertEqual(self.test_out.getvalue().strip(),
                                 "(10.00%) [=====>                                             ]")
            elif downloaded_bytes == 7:
                self.assertIn("(70.00%) [===================================>               ]",
                              self.test_out.getvalue().strip())
            elif downloaded_bytes == 10:
                self.assertIn("(100.00%) [==================================================>]",
                              self.test_out.getvalue().strip())

        self.assertStdoutContains(["[", ">", "=", "]"])

    @patch("builtins.open", new_callable=mock_open, read_data="a sentence")
    @patch("deepparse.fasttext_tools.urlopen")
    @patch("deepparse.fasttext_tools.gzip")
    @patch("deepparse.fasttext_tools.os.rename")
    @patch("deepparse.fasttext_tools.shutil")
    @patch("deepparse.fasttext_tools.os.remove")
    def test_givenADownloadFasttext_whenPrintProgressSetToVerbose_thenDontPrint(self, os_remove_mock, shutil_mock,
                                                                                os_rename_mock, g_zip_mock,
                                                                                urlopen_mock, open_mock):
        # pylint: disable=too-many-arguments
        urlopen_mock().read.side_effect = self.a_response_payload
        self._capture_output()
        with urlopen_mock:
            _ = download_fasttext_embeddings(self.a_directory_path, verbose=False)

        expected = ""
        actual = self.test_out.getvalue().strip()

        self.assertEqual(expected, actual)

    @patch("builtins.open", new_callable=mock_open, read_data="a sentence")
    @patch("deepparse.fasttext_tools.urlopen")
    @patch("deepparse.fasttext_tools.gzip")
    @patch("deepparse.fasttext_tools.os.rename")
    @patch("deepparse.fasttext_tools.shutil")
    @patch("deepparse.fasttext_tools.os.remove")
    def test_givenADownloadFasttext_whenPrintProgressSetToVerbose_thenPrint(self, os_remove_mock, shutil_mock,
                                                                            os_rename_mock, g_zip_mock, urlopen_mock,
                                                                            open_mock):
        # pylint: disable=too-many-arguments
        urlopen_mock().read.side_effect = self.a_response_payload
        urlopen_mock().getheader.return_value = "2"
        self._capture_output()
        with urlopen_mock:
            _ = download_fasttext_embeddings(self.a_directory_path, verbose=True)
        actual = self.test_out.getvalue().strip()

        expected = "The fastText pre-trained word embeddings will be downloaded (6.8 GO), " \
                   "this process will take several minutes."
        self.assertIn(expected, actual)

        expected = "Downloading https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"
        self.assertIn(expected, actual)

        expected = "(50.00%) [=========================>                         ]"
        self.assertIn(expected, actual)

        expected = "(100.00%) [==================================================>]"
        self.assertIn(expected, actual)

    def test_givenAFasttextEmbeddingsToLoad_whenLoad_thenLoadProperly(self):
        download_from_url("fake_embeddings_cc.fr.300", self.a_directory_path, "bin")
        embeddings_path = self.a_fake_embeddings_path

        embeddings = load_fasttext_embeddings(embeddings_path)

        self.assertIsInstance(embeddings, _FastText)


if __name__ == "__main__":
    unittest.main()
