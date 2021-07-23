import os
import platform
from tempfile import TemporaryDirectory
from unittest import skipIf
from unittest.mock import patch

from fasttext.FastText import _FastText
from gensim.models.fasttext import FastTextKeyedVectors
from torch.utils.data import DataLoader

from deepparse import download_from_url
from deepparse.embeddings_models import FastTextEmbeddingsModel
from tests.embeddings_models.integration.tools import MockedDataTransform
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


class FastTextEmbeddingsModelIntegrationTest(AddressParserRetrainTestCase):

    @classmethod
    def setUpClass(cls):
        super(FastTextEmbeddingsModelIntegrationTest, cls).setUpClass()
        cls.file_name = "fake_embeddings_cc.fr.300"
        cls.temp_dir_obj = TemporaryDirectory()
        cls.fake_cache_path = os.path.join(cls.temp_dir_obj.name, "fake_cache")
        download_from_url(cls.file_name, cls.fake_cache_path, "bin")

        cls.a_fasttext_model_path = os.path.join(cls.fake_cache_path, cls.file_name + ".bin")

        cls.verbose = False

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelInit_thenLoadWithProperFunction(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        self.assertIsInstance(model.model, FastTextKeyedVectors)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenFasttextModelInit_thenLoadWithProperFunction(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        self.assertIsInstance(model.model, _FastText)

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
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
