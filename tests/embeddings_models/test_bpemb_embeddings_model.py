import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import requests

from deepparse.embeddings_models import BPEmbEmbeddingsModel


class BPEmbEmbeddingsModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_path = "."
        cls.a_word = "test"
        cls.dim = 9

    def setUp(self):
        self.model = MagicMock()
        self.model.dim = self.dim

    def test_whenInstantiatedWithPath_thenShouldLoadBPEmbModel(self):
        with patch(
            "deepparse.embeddings_models.bpemb_embeddings_model.BPEmbBaseURLWrapperBugFix",
            return_value=self.model,
        ) as loader:
            _ = BPEmbEmbeddingsModel(self.a_path, verbose=False)

            loader.assert_called_with(lang="multi", vs=100000, dim=300, cache_dir=Path(self.a_path))

    def test_whenInstantiatedAndDownloadSucceeds_thenSSLVerificationIsNotDisabled(self):
        # The default path must download with SSL verification ON: no_ssl_verification must not be entered.
        with patch(
            "deepparse.embeddings_models.bpemb_embeddings_model.BPEmbBaseURLWrapperBugFix",
            return_value=self.model,
        ):
            with patch("deepparse.embeddings_models.bpemb_embeddings_model.no_ssl_verification") as no_ssl_mock:
                BPEmbEmbeddingsModel(self.a_path, verbose=False)

            no_ssl_mock.assert_not_called()

    def test_whenDownloadRaisesSSLError_thenFailsafeRetriesWithoutSSLVerification(self):
        # Failsafe: on an SSL error, retry once inside no_ssl_verification and still return a working model.
        with patch(
            "deepparse.embeddings_models.bpemb_embeddings_model.BPEmbBaseURLWrapperBugFix",
            side_effect=[requests.exceptions.SSLError("broken certificate"), self.model],
        ) as loader:
            with patch("deepparse.embeddings_models.bpemb_embeddings_model.no_ssl_verification") as no_ssl_mock:
                embeddings_model = BPEmbEmbeddingsModel(self.a_path, verbose=False)

            no_ssl_mock.assert_called_once()
            self.assertEqual(loader.call_count, 2)
            self.assertIs(embeddings_model.model, self.model)

    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch(
            "deepparse.embeddings_models.bpemb_embeddings_model.BPEmbBaseURLWrapperBugFix",
            return_value=self.model,
        ):
            embeddings_model = BPEmbEmbeddingsModel(self.a_path, verbose=False)

            embeddings_model(self.a_word)

            self.model.embed.assert_called_with(self.a_word)

    def test_givenADimOf9_whenAskDimProperty_thenReturnProperDim(self):
        with patch(
            "deepparse.embeddings_models.bpemb_embeddings_model.BPEmbBaseURLWrapperBugFix",
            return_value=self.model,
        ):
            embeddings_model = BPEmbEmbeddingsModel(self.a_path, verbose=False)

            actual = embeddings_model.dim
            expected = self.dim
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
