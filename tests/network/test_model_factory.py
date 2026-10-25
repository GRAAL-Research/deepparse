import unittest
import warnings
from unittest import TestCase
from unittest.mock import Mock

from deepparse.network import BPEmbSeq2SeqModel, FastTextSeq2SeqModel, Model2VecSeq2SeqModel, ModelFactory


class ModelFactoryTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_fasttext_model_type = "fasttext"
        cls.a_bpemb_model_type = "bpemb"
        cls.an_invalid_model_type = "invalid_model"

        cls.a_cache_dir = "~/.cache/deepparse"
        cls.a_device = "cpu"

    def setUp(self):
        self.loader = Mock()
        a_model_version = "version"
        self.loader.load_pre_trained_model.side_effect = lambda *args: (args[0], a_model_version)

        self.factory = ModelFactory(self.loader)

    def test_givenAFasttextModelType_whenCreatingModel_thenShouldReturnFasttextSeq2Seq(self):
        model, _ = self.factory.create(self.a_fasttext_model_type, self.a_device)

        self.assertIsInstance(model, FastTextSeq2SeqModel)

    def test_givenABpembModelType_whenCreatingModel_thenShouldReturnBpembSeq2Seq(self):
        model, _ = self.factory.create(self.a_bpemb_model_type, self.a_device)

        self.assertIsInstance(model, BPEmbSeq2SeqModel)

    def test_givenModel2VecModelTypes_whenCreatingModel_thenShouldUseTheEmbeddingsDimensionAsInputSize(self):
        for model_type, expected_input_size in (
            ("model2vec-8m", 256),
            ("model2vec-32m_attention", 512),
            ("model2vec-multilingual", 256),
        ):
            with self.subTest(model_type=model_type):
                model, _ = self.factory.create(model_type, self.a_device, pre_trained_weights=False)

                self.assertIsInstance(model, Model2VecSeq2SeqModel)
                self.assertEqual(expected_input_size, model.encoder.lstm.input_size)

    def test_givenAModel2VecRetrainedInputSize_whenCreatingModel_thenShouldKeepTheRetrainedInputSize(self):
        model, _ = self.factory.create("model2vec-8m", self.a_device, pre_trained_weights=False, input_size=300)

        self.assertEqual(300, model.encoder.lstm.input_size)

    def test_givenAModelTypeWithoutPublishedWeights_whenCreatingPreTrainedModel_thenWarnAndDoNotDownload(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _, version = self.factory.create("model2vec-8m", self.a_device, pre_trained_weights=True)

        self.assertTrue(any("no pre-trained weights" in str(warning.message) for warning in caught_warnings))

        self.loader.load_pre_trained_model.assert_not_called()
        self.assertEqual("", version)

    def test_givenAnInvalidModelType_whenCreatingModel_thenShouldRaiseException(self):
        with self.assertRaises(NotImplementedError):
            self.factory.create(self.an_invalid_model_type, self.a_device)


if __name__ == "__main__":
    unittest.main()
