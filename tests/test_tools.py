import os
from unittest import TestCase

import requests

from deepparse import download_from_url


class ToolsTests(TestCase):
    def setUp(self) -> None:
        self._a_saving_dir = "./tmp"
        self._a_file_extension = "version"

    def test_givenFasttextVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "fasttext"

        download_from_url(file_name, self._a_saving_dir, self._a_file_extension)

        self.assertTrue(os.path.exists(os.path.join(self._a_saving_dir, f"{file_name}.{self._a_file_extension}")))

    def test_givenFasttextVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_fasttext"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_url(wrong_file_name, self._a_saving_dir, self._a_file_extension)

    def test_givenBPEmbVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "bpemb"

        download_from_url(file_name, self._a_saving_dir, self._a_file_extension)

        self.assertTrue(os.path.exists(os.path.join(self._a_saving_dir, f"{file_name}.{self._a_file_extension}")))

    def test_givenBPEmbVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_bpemb"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_url(wrong_file_name, self._a_saving_dir, self._a_file_extension)
