# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import platform
from tempfile import TemporaryDirectory
from unittest import skipIf
from unittest.mock import patch

from fasttext.FastText import _FastText
from gensim.models import FastText
from gensim.test.utils import common_texts
from gensim.models._fasttext_bin import save
from gensim.models.fasttext import FastTextKeyedVectors
from torch.utils.data import DataLoader

from deepparse.embeddings_models import FastTextEmbeddingsModel
from tests.embeddings_models.integration.tools import MockedDataTransform
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


#@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class FastTextEmbeddingsModelIntegrationTest(AddressParserRetrainTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastTextEmbeddingsModelIntegrationTest, cls).setUpClass()

        # We create and save a dummy fasttext embeddings model
        cls.file_name = "fake_embeddings_cc.fr.300.bin"
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_fasttext_model_path = os.path.join(cls.temp_dir_obj.name, cls.file_name)

        model = FastText(vector_size=4, window=3, min_count=1)
        model.build_vocab(corpus_iterable=common_texts)
        model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=1)
        save(model, cls.a_fasttext_model_path, {"lr_update_rate": 100, "word_ngrams": 1}, "utf-8")

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
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoader_thenWorkProperly(
        self,
    ):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
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
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoaderNumWorkers2_thenWorkProperly(
        self,
    ):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
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
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_givenUbuntu_whenFasttextModelCollateFnInDataLoaderEvenWithWindowsSetup_thenWorkProperly(
        self, platform_mock
    ):
        platform_mock.system().__eq__.return_value = False
        with platform_mock:
            model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
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
    @patch("deepparse.embeddings_models.fasttext_embeddings_model.platform")
    def test_givenUbuntu_whenFasttextModelCollateFnInDataLoaderForWindowsNumWorkers2_thenWorkProperly(
        self, platform_mock
    ):
        platform_mock.system().__eq__.return_value = False
        with platform_mock:
            model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

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
