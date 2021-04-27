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
        cls.dim = 9

    def setUp(self):
        model = MagicMock()
        shape_mock = MagicMock()
        shape_mock.return_value = self.dim
        model.get_dimension = shape_mock
        self.model = model

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_whenInstantiatedWithPath_thenShouldLoadFasttextModel(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model) as loader:
            _ = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            loader.assert_called_with(self.a_path)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_whenCalledToEmbed_thenShouldCallLoadedModel(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model):
            embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            embeddings_model(self.a_word)

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

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_givenADimOf9Windows_whenAskDimProperty_thenReturnProperDim(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_facebook_vectors",
                   return_value=self.model):
            embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            actual = embeddings_model.dim
            expected = self.dim
            self.assertEqual(actual, expected)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenADimOf9Linux_whenAskDimProperty_thenReturnProperDim(self):
        with patch("deepparse.embeddings_models.fasttext_embeddings_model.load_fasttext_embeddings",
                   return_value=self.model):
            embeddings_model = FastTextEmbeddingsModel(self.a_path, verbose=self.verbose)

            actual = embeddings_model.dim
            expected = self.dim
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
