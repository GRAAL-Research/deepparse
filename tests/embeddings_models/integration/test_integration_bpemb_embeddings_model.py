# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import platform
from tempfile import TemporaryDirectory
from unittest import skipIf

from bpemb import BPEmb
from torch.utils.data import DataLoader

from deepparse.embeddings_models import BPEmbEmbeddingsModel
from tests.base_file_exist import FileCreationTestCase
from tests.embeddings_models.integration.tools import MockedDataTransform
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class BPEmbEmbeddingsModelIntegrationTest(AddressParserRetrainTestCase, FileCreationTestCase):
    @classmethod
    def setUpClass(cls):
        super(BPEmbEmbeddingsModelIntegrationTest, cls).setUpClass()
        cls.temp_dir_obj = TemporaryDirectory()
        cls.fake_cache_path = os.path.join(cls.temp_dir_obj.name, "fake_cache")

        cls.verbose = False

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenANewCacheDir_whenBPEmbModelInit_thenCreateNewCache(self):
        BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)

        self.assertFileExist(os.path.join(self.fake_cache_path, "multi"))

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenBPEmbModelInit_thenLoadWithProperFunction(self):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)

        self.assertIsInstance(model.model, BPEmb)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenBPEmbModelInit_thenLoadWithProperFunction(self):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)

        self.assertIsInstance(model.model, BPEmb)

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenBPEmbModelCollateFnInDataLoader_thenWorkProperly(self):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=0,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenBPEmbModelCollateFnInDataLoaderNumWorkers1_thenWorkProperly(
        self,
    ):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=1,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenBPEmbModelCollateFnInDataLoaderNumWorkers2_thenWorkProperly(
        self,
    ):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=2,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenBPEmbModelCollateFnInDataLoaderForWindows_thenWorkProperly(
        self,
    ):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=0,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenBPEmbModelCollateFnInDataLoaderForWindowsNumWorkers1_thenWorkProperly(
        self,
    ):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=1,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    @skipIf(platform.system() == "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenBPEmbModelCollateFnInDataLoaderForWindowsNumWorkers2_thenWorkProperly(
        self,
    ):
        model = BPEmbEmbeddingsModel(self.fake_cache_path, verbose=self.verbose)

        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=2,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)
