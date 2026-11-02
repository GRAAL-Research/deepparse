# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import skipIf

from torch.utils.data import DataLoader

from deepparse.embeddings_models import FastTextBinReader, FastTextEmbeddingsModel
from tests.embeddings_models.integration.tools import MockedDataTransform
from tests.fixtures.fasttext_bin_writer import create_tiny_fasttext_bin
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class FastTextEmbeddingsModelIntegrationTest(AddressParserRetrainTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastTextEmbeddingsModelIntegrationTest, cls).setUpClass()

        # We create and save a dummy fasttext embeddings model
        cls.file_name = "fake_embeddings_cc.fr.300.bin"
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_fasttext_model_path = os.path.join(cls.temp_dir_obj.name, cls.file_name)

        create_tiny_fasttext_bin(cls.a_fasttext_model_path)

        cls.verbose = False

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def test_whenFasttextModelInit_thenLoadWithTheBinReader(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        self.assertIsInstance(model.model, FastTextBinReader)

    def assert_collate_fn_works_in_data_loader(self, num_workers: int, multiprocessing_context=None) -> None:
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)
        data_transform = MockedDataTransform(model)

        data_loader = DataLoader(
            self.training_container,
            collate_fn=data_transform.collate_fn,
            batch_size=32,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
        )
        dataset = []
        for data in data_loader:
            dataset.append(data)
        self.assertGreater(len(dataset), 0)

    def test_whenFasttextModelCollateFnInDataLoader_thenWorkProperly(self):
        self.assert_collate_fn_works_in_data_loader(num_workers=0)

    def test_whenFasttextModelCollateFnInDataLoaderNumWorkers2_thenWorkProperly(self):
        self.assert_collate_fn_works_in_data_loader(num_workers=2)

    def test_whenFasttextModelCollateFnInDataLoaderWithSpawnWorkers_thenWorkProperly(self):
        # Windows and MacOS start the workers with 'spawn', which pickles the embeddings model.
        self.assert_collate_fn_works_in_data_loader(num_workers=2, multiprocessing_context="spawn")
