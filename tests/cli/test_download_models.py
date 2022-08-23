# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, call

from deepparse.cli import download_models


class DownloadModelTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_dir = os.path.join(self.temp_dir_obj.name, "fake_cache")
        self.models_type = ["fasttext", "fasttext_attention", "bpemb", "bpemb_attention"]

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    @patch("deepparse.cli.download_models.download_weights")
    @patch("deepparse.cli.download_models.download_fasttext_embeddings")
    @patch("deepparse.cli.download_models.download_fasttext_magnitude_embeddings")
    @patch("deepparse.cli.download_models.BPEmb")
    def test_givenADownloadAllModels_whenModelsAreNotLocal_thenDownloadAllModels(
        self,
        BPEmb_mock,
        download_fasttext_magnitude_embeddings_mock,
        download_fasttext_embeddings_mock,
        weights_download_mock,
    ):
        download_models.main(["--saving_cache_dir", self.fake_cache_dir])

        download_fasttext_embeddings_mock.assert_called()
        download_fasttext_embeddings_mock.assert_called_with(cache_dir=self.fake_cache_dir)

        download_fasttext_magnitude_embeddings_mock.assert_called()
        download_fasttext_magnitude_embeddings_mock.assert_called_with(cache_dir=self.fake_cache_dir)

        BPEmb_mock.assert_called()
        BPEmb_mock.assert_called_with(lang="multi", vs=100000, dim=300, cache_dir=self.fake_cache_dir)

        weights_download_mock.assert_called()

        expected_call_count = 4
        actual_call_count = weights_download_mock.call_count
        self.assertEqual(expected_call_count, actual_call_count)

        for model_type in self.models_type:
            weights_download_mock.assert_has_calls([call(model_type, saving_dir=self.fake_cache_dir)])


if __name__ == "__main__":
    unittest.main()
