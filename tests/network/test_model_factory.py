import unittest
from unittest import TestCase

from deepparse.network import ModelFactory, FastTextSeq2SeqModel, BPEmbSeq2SeqModel


class ModelFactoryTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_fasttext_model_type = "fasttext"
        cls.a_bpemb_model_type = "bpemb"
        cls.an_invalid_model_type = "invalid_model"

        cls.a_cache_dir = "~/.cache/deepparse"
        cls.a_device = "cpu"

    def setUp(self):
        self.factory = ModelFactory()

    def test_givenAFasttextModelType_whenCreatingModel_thenShouldReturnFasttextSeq2Seq(self):
        model = self.factory.create(self.a_fasttext_model_type, self.a_cache_dir, self.a_device)

        self.assertIsInstance(model, FastTextSeq2SeqModel)

    def test_givenABpembModelType_whenCreatingModel_thenShouldReturnBpembSeq2Seq(self):
        model = self.factory.create(self.a_bpemb_model_type, self.a_cache_dir, self.a_device)

        self.assertIsInstance(model, BPEmbSeq2SeqModel)

    def test_givenAnInvalidModelType_whenCreatingModel_thenShouldRaiseException(self):
        with self.assertRaises(NotImplementedError):
            self.factory.create(self.an_invalid_model_type, self.a_cache_dir, self.a_device)


if __name__ == "__main__":
    unittest.main()
