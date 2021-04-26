import os
import platform
from unittest import skipIf

from fasttext.FastText import _FastText
from gensim.models.fasttext import FastTextKeyedVectors
from torch.utils.data import DataLoader

from deepparse.embeddings_models import FastTextEmbeddingsModel
from tests.parser.integration.base_integration import AddressParserIntegrationTestCase


class AddressParserPredictCPUTest(AddressParserIntegrationTestCase):

    def setUp(self):
        self.a_fasttext_model_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")
        self.verbose = False

    @skipIf(not platform.system() == "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelInit_thenLoadWithProperFunction(self):
        model = FastTextEmbeddingsModel("", verbose=self.verbose)

        self.assertIsInstance(model.model, FastTextKeyedVectors)

    @skipIf(not platform.system() != "Windows", "Integration test not on Windows env.")
    def test_givenANotWindowsOS_whenFasttextModelInit_thenLoadWithProperFunction(self):
        model = FastTextEmbeddingsModel(self.a_fasttext_model_path, verbose=self.verbose)

        self.assertIsInstance(model.model, _FastText)

    @skipIf(not platform.system() != "Windows", "Integration test on Windows env.")
    def test_givenAWindowsOS_whenFasttextModelCollateFnInDataLoader_thenWorkProperly(self):
        model = FastTextEmbeddingsModel("", verbose=self.verbose)

        data_loader = DataLoader(self.training_container,
                                 collate_fn=model,
                                 batch_size=32,
                                 num_workers=0,
                                 shuffle=True)

        self.assertIsInstance(model.model, FastTextKeyedVectors)
