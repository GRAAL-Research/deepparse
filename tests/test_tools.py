import os
from unittest import TestCase

import requests

from deepparse import download_from_url, latest_version


class ToolsTests(TestCase):

    def setUp(self) -> None:
        self.a_saving_dir = "./tmp"
        self.a_file_extension = "version"

        self.fake_cache_path = "./"
        self.latest_fasttext_version = "617a417a2f2b02654f7deb5b5cbc60ab2f6334ba"
        self.latest_bpemb_version = "6d01367745157066ea6e621ac087be828137711f"

    def create_cache_version(self, model_name, content):
        file = open(os.path.join(self.fake_cache_path, model_name + ".version"), "w")
        file.write(content)
        file.close()

    def test_givenAFasttextLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version('fasttext', self.latest_fasttext_version)
        self.assertTrue(latest_version('fasttext', self.fake_cache_path))

    def test_givenAFasttextNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(self):
        self.create_cache_version('fasttext', "not_the_last_version")
        self.assertFalse(latest_version('fasttext', self.fake_cache_path))

    def test_givenABPEmbLatestVersion_whenVerifyIfLastVersion_thenReturnTrue(self):
        self.create_cache_version('bpemb', self.latest_bpemb_version)
        self.assertTrue(latest_version('bpemb', self.fake_cache_path))

    def test_givenABPEmbNotTheLatestVersion_whenVerifyIfLastVersion_thenReturnFalse(self):
        self.create_cache_version('bpemb', "not_the_last_version")
        self.assertFalse(latest_version('bpemb', self.fake_cache_path))

    def test_givenFasttextVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "fasttext"

        download_from_url(file_name, self.a_saving_dir, self.a_file_extension)

        self.assertTrue(os.path.exists(os.path.join(self.a_saving_dir, f"{file_name}.{self.a_file_extension}")))

    def test_givenFasttextVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_fasttext"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_url(wrong_file_name, self.a_saving_dir, self.a_file_extension)

    def test_givenBPEmbVersion_whenDownloadOk_thenDownloadIt(self):
        file_name = "bpemb"

        download_from_url(file_name, self.a_saving_dir, self.a_file_extension)

        self.assertTrue(os.path.exists(os.path.join(self.a_saving_dir, f"{file_name}.{self.a_file_extension}")))

    def test_givenBPEmbVersion_whenDownload404_thenHTTPError(self):
        wrong_file_name = "wrong_bpemb"

        with self.assertRaises(requests.exceptions.HTTPError):
            download_from_url(wrong_file_name, self.a_saving_dir, self.a_file_extension)
