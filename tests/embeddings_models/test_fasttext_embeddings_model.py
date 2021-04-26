import platform
import unittest
from unittest import TestCase, skipIf
from unittest.mock import patch, MagicMock

from deepparse.embeddings_models import FastTextEmbeddingsModel


class FasttextEmbeddingsModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_path = "."
        cls.a_word = "test"
        cls.verbose = False

    def setUp(self):
        self.model = MagicMock()
        self.model.dim = 9

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_whenInstantiatedWithPath_thenShouldLoadFasttextModel(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model) as loader:
            self.embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            loader.assert_called_with(self.a_path)

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model):
            self.embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            self.embeddings_model(self.a_word)

            self.model.__getitem__.assert_called_with(self.a_word)

    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_whenInstantiatedOnWindows_thenShouldLoadFasttextModelGensim(self, platform_mock):
        platform_mock.system().__eq__.return_value = True
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_facebook_vectors",
                   return_value=self.model) as loader:
            with platform_mock:
                self.embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

                loader.assert_called_with(self.a_path)

    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_whenInstantiatedOnLinux_thenShouldLoadFasttextModel(self, platform_mock):
        platform_mock.system().__eq__.return_value = False
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model) as loader:
            with platform_mock:
                self.embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

                loader.assert_called_with(self.a_path)

    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_whenInstantiatedOnMacOS_thenShouldLoadFasttextModel(self, platform_mock):
        platform_mock.system().__eq__.return_value = False
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model) as loader:
            with platform_mock:
                self.embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

                loader.assert_called_with(self.a_path)


if __name__ == "__main__":
    unittest.main()
