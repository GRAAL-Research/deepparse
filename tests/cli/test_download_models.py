# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, patch

from deepparse.cli.download_models import main as download_models_cli_main


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
    @patch("deepparse.download_tools.BPEmbLoader")
    def test_givenADownloadAllModels_whenModelsAreNotLocal_thenDownloadAllModels(
        self,
        BPEmb_mock,
        download_fasttext_magnitude_embeddings_mock,
        download_fasttext_embeddings_mock,
        download_model2vec_embeddings_mock,
        weights_download_mock,
    ):
        download_models_cli_main(["--saving_cache_dir", self.fake_cache_dir])

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


if __name__ == "__main__":
    unittest.main()
