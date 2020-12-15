# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=W0613

import argparse
import os
from unittest import TestCase
from unittest.mock import patch

from deepparse import download
from tests.tools import create_file


class DownloadTests(TestCase):

    def setUp(self) -> None:
        self.fake_cache_path = "./"
        self.a_file_extension = "version"
        self.a_fasttext_model_type = "fasttext"
        self.a_fasttext_light_model_type = "fasttext-light"
        self.a_bpemb_model_type = "bpemb"
        self.latest_fasttext_version = "617a417a2f2b02654f7deb5b5cbc60ab2f6334ba"
        self.latest_bpemb_version = "6d01367745157066ea6e621ac087be828137711f"

        self.create_parser()

    def tearDown(self) -> None:
        if os.path.exists("fasttext.version"):
            os.remove("fasttext.version")
        if os.path.exists("bpemb.version"):
            os.remove("bpemb.version")

    def create_cache_version(self, model_name, content):
        version_file_path = os.path.join(self.fake_cache_path, model_name + ".version")
        create_file(version_file_path, content)

    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "model_type",
            choices=[self.a_fasttext_model_type, self.a_fasttext_light_model_type, self.a_bpemb_model_type])

    @patch("deepparse.download.download_weights")
    @patch("deepparse.download.CACHE_PATH", "./")  # we mock the cache path with the fake_cache_path
    def test_givenAFasttextDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        args_parser = self.parser.parse_args([self.a_fasttext_model_type])

        with patch("deepparse.download.download_fasttext_embeddings") as downloader:
            download.main(args_parser)

            downloader.assert_any_call(saving_dir=self.fake_cache_path)

    @patch("deepparse.download.download_weights")
    @patch("deepparse.download.CACHE_PATH", "./")  # we mock the cache path with the fake_cache_path
    def test_givenAFasttextMagnitudeDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        args_parser = self.parser.parse_args([self.a_fasttext_light_model_type])

        with patch("deepparse.download.download_fasttext_magnitude_embeddings") as downloader:
            download.main(args_parser)

            downloader.assert_any_call(saving_dir=self.fake_cache_path)

    @patch("deepparse.download.download_weights")
    @patch("deepparse.download.CACHE_PATH", "./")  # we mock the cache path with the fake_cache_path
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        args_parser = self.parser.parse_args([self.a_bpemb_model_type])

        with patch("deepparse.download.BPEmb") as downloader:
            download.main(args_parser)

            downloader.assert_any_call(lang="multi", vs=100000, dim=300)  # settings for BPemb

    @patch("deepparse.download.download_fasttext_embeddings")
    @patch("deepparse.download.CACHE_PATH", "./")  # we mock the cache path with the fake_cache_path
    def test_givenAFasttextDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        args_parser = self.parser.parse_args([self.a_fasttext_model_type])

        with patch("deepparse.download.download_weights") as downloader:
            download.main(args_parser)

            downloader.assert_any_call(self.a_fasttext_model_type, self.fake_cache_path)

    @patch("deepparse.download.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download.CACHE_PATH", "./")  # we mock the cache path with the fake_cache_path
    def test_givenAFasttextLightDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        args_parser = self.parser.parse_args([self.a_fasttext_light_model_type])

        with patch("deepparse.download.download_weights") as downloader:
            download.main(args_parser)

            downloader.assert_any_call(self.a_fasttext_light_model_type, self.fake_cache_path)

    @patch("deepparse.download.BPEmb")
    @patch("deepparse.download.CACHE_PATH", "./")  # we mock the cache path with the fake_cache_path
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        args_parser = self.parser.parse_args([self.a_bpemb_model_type])

        with patch("deepparse.download.download_weights") as downloader:
            download.main(args_parser)

            downloader.assert_any_call(self.a_bpemb_model_type, self.fake_cache_path)
