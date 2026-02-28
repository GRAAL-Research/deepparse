import unittest
from unittest import TestCase
from unittest.mock import Mock

from deepparse.network import BPEmbSeq2SeqModel, FastTextSeq2SeqModel, ModelFactory


class ModelFactoryTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_fasttext_model_type = "fasttext"
        cls.a_bpemb_model_type = "bpemb"
        cls.an_invalid_model_type = "invalid_model"

        cls.a_cache_dir = "~/.cache/deepparse"
        cls.a_device = "cpu"

    def setUp(self):
        loader = Mock()
        a_model_version = "version"
        loader.load_pre_trained_model.side_effect = lambda *args: (args[0], a_model_version)

        self.factory = ModelFactory(loader)

    def test_givenAFasttextModelType_whenCreatingModel_thenShouldReturnFasttextSeq2Seq(self):
        model, _ = self.factory.create(self.a_fasttext_model_type, self.a_device)

        self.assertIsInstance(model, FastTextSeq2SeqModel)

    def test_givenABpembModelType_whenCreatingModel_thenShouldReturnBpembSeq2Seq(self):
        model, _ = self.factory.create(self.a_bpemb_model_type, self.a_device)

        self.assertIsInstance(model, BPEmbSeq2SeqModel)

    def test_givenAnInvalidModelType_whenCreatingModel_thenShouldRaiseException(self):
        with self.assertRaises(NotImplementedError):
            self.factory.create(self.an_invalid_model_type, self.a_device)


if __name__ == "__main__":
    unittest.main()
