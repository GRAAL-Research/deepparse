# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=W0613

import gzip
import io
import os
import sys
import unittest
from unittest import TestCase
from unittest.mock import patch

from deepparse import download_fasttext_embeddings
from deepparse.fasttext_tools import _print_progress
from tests.tools import create_file


class ToolsTests(TestCase):

    def setUp(self) -> None:
        self.a_directory_path = "./"
        self.a_fasttext_file_name_path = os.path.join(self.a_directory_path, "cc.fr.300.bin")
        self.a_fasttext_gz_file_name_path = os.path.join(self.a_directory_path, "cc.fr.300.bin.gz")

    def tearDown(self) -> None:
        if os.path.exists(self.a_fasttext_file_name_path):
            os.remove(self.a_fasttext_file_name_path)
        if os.path.exists(self.a_fasttext_gz_file_name_path):
            os.remove(self.a_fasttext_gz_file_name_path)

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out

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

        with patch("deepparse.fasttext_tools.download_gz_model") as _:
            actual = download_fasttext_embeddings(self.a_directory_path)
            expected = self.a_fasttext_file_name_path
            self.assertEqual(actual, expected)

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


if __name__ == "__main__":
    unittest.main()
