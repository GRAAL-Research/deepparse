# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from deepparse.cli.download_model import main as download_model_cli_main


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
                download_model_cli_main([self.a_fasttext_model_type])

                downloader.assert_called()
                downloader.assert_called_with(cache_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_weights")
    def test_givenADownload_whenCachePathIsChange_thenDownloadInCacheDir(self, weights_download_mock):
        with patch("deepparse.download_tools.download_fasttext_embeddings") as downloader:
            download_model_cli_main([self.a_fasttext_model_type, "--saving_cache_dir", self.fake_cache_dir])

            downloader.assert_called()
            downloader.assert_called_with(cache_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_weights")
    def test_givenAFasttextMagnitudeDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_fasttext_magnitude_embeddings") as downloader:
                download_model_cli_main([self.a_fasttext_light_model_type])

                downloader.assert_called()
                downloader.assert_called_with(cache_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_weights")
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadEmbeddings(self, weights_download_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.BPEmb") as downloader:
                download_model_cli_main([self.a_bpemb_model_type])

                downloader.assert_called()
                downloader.assert_any_call(
                    lang="multi", vs=100000, dim=300, cache_dir=self.fake_cache_dir
                )  # settings for BPemb

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    def test_givenAFasttextDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_fasttext_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    def test_givenAFasttextAttDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_fasttext_att_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_att_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    def test_givenAFasttextLightDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_fasttext_light_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_light_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    def test_givenABPembDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_bpemb_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    def test_givenABPembAttDownload_whenModelIsNotLocal_thenDownloadWeights(self, download_embeddings_mock):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_bpemb_att_model_type])

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
                download_model_cli_main([self.a_fasttext_model_type])

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
                download_model_cli_main([self.a_fasttext_light_model_type])

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
                download_model_cli_main([self.a_bpemb_model_type])

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
                download_model_cli_main([self.a_fasttext_model_type])

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
                download_model_cli_main([self.a_fasttext_light_model_type])

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
                download_model_cli_main([self.a_bpemb_model_type])

                downloader.assert_not_called()

    @patch("deepparse.download_tools.download_fasttext_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenAFasttextDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_fasttext_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_model_type, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.download_fasttext_magnitude_embeddings")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenAFasttextLightDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_fasttext_light_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_fasttext_light_model_file_name, saving_dir=self.fake_cache_dir)

    @patch("deepparse.download_tools.BPEmb")
    @patch("deepparse.download_tools.os.path.isfile", side_effect=[False, True])  # no version file in local
    def test_givenABPembDownload_whenModelIsNotLocalButNotLatest_thenDownloadWeights(
        self, download_embeddings_mock, os_is_file_mock
    ):
        with patch("deepparse.download_tools.CACHE_PATH", self.fake_cache_dir):
            with patch("deepparse.download_tools.download_weights") as downloader:
                download_model_cli_main([self.a_bpemb_model_type])

                downloader.assert_called()
                downloader.assert_any_call(self.a_bpemb_model_type, saving_dir=self.fake_cache_dir)


if __name__ == "__main__":
    unittest.main()
