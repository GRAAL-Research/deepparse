import os
import platform
from unittest import skipIf
from unittest.mock import patch

from fasttext.FastText import _FastText
from gensim.models.fasttext import FastTextKeyedVectors
from torch.utils.data import DataLoader

from deepparse import download_from_url
from deepparse.embeddings_models import FastTextEmbeddingsModel
from tests.parser.integration.base_integration import AddressParserIntegrationTestCase


# A class to mock to logic of using a collate_fn in the data loader to be multiprocess and test if multiprocess work
class MockedDataTransform:

    def __init__(self, word_vectors_model):
        self.model = word_vectors_model

    def collate_fn(self, x):
        words = []
        for data_sample in x:
            for word in data_sample[0].split():
                words.append(self.model(word))
        return words


class FastTextEmbeddingsModelIntegrationTest(AddressParserIntegrationTestCase):

    @classmethod
    def setUpClass(cls):
        super(FastTextEmbeddingsModelIntegrationTest, cls).setUpClass()
        cls.file_name = "fake_embeddings_cc.fr.300"
        cls.fake_cache_path = "./"
        download_from_url(cls.file_name, cls.fake_cache_path, "bin")

        cls.a_fasttext_model_path = os.path.join(cls.fake_cache_path, cls.file_name + ".bin")

        cls.verbose = False

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.a_fasttext_model_path):
            os.remove(cls.a_fasttext_model_path)

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelInit_thenLoadWithProperFunction(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        self.assertIsInstance(model.model, FastTextKeyedVectors)

    @skipIf(not platform.system() != "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenFasttextModelInit_thenLoadWithProperFunction(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        self.assertIsInstance(model.model, _FastText)

    @skipIf(not platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoader_thenWorkProperly(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=0)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderNumWorkers1_thenWorkProperly(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=1)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderNumWorkers2_thenWorkProperly(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=2)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() != "Windows", "Integration test on Windows env.")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderEvenWithWindowsSetup_thenWorkProperly(
            self, platform_mock):
        platform_mock.system().__eq__.return_value = True
        with platform_mock:
            model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=0)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderForWindows_thenRaiseError(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=0)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderForWindowsNumWorkers1_thenRaiseError(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=1)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderForWindowsNumWorkers2_thenRaiseError(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=2)
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderForNotWindows_thenRaiseError(self, platform_mock):
        platform_mock.system().__eq__.return_value = True
        with platform_mock:
            model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=data_transform.collate_fn,
                                 batch_size=32,
                                 num_workers=0)
        with self.assertRaises(TypeError):
            for _ in data_loader:
                pass
