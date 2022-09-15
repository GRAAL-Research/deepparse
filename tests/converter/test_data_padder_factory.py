import unittest
from unittest import TestCase

from deepparse.converter import DataPadderFactory, WordEmbeddingPadder, SubwordEmbeddingPadder


class DataPadderFactoryTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_padding_value = -100
        cls.a_bpemb_embedding_model_type = "bpemb"
        cls.a_fasttext_embedding_model_type = "fasttext"
        cls.a_magnitude_embedding_model_type = "fasttext_magnitude"

    def setUp(self):
        self.padder_factory = DataPadderFactory()

    def test_givenAFasttextEmbeddingModelType_whenCreatingPadder_thenShouldReturnCorrecttPadder(self):
        padder = self.padder_factory.create(self.a_fasttext_embedding_model_type, self.a_padding_value)

        self.assertIsInstance(padder, WordEmbeddingPadder)

    def test_givenAMagnitudeEmbeddingModelType_whenCreatingPadder_thenShouldReturnCorrecttPadder(self):
        padder = self.padder_factory.create(self.a_magnitude_embedding_model_type, self.a_padding_value)

        self.assertIsInstance(padder, WordEmbeddingPadder)

    def test_givenABpembEmbeddingModelType_whenCreatingPadder_thenShouldReturnCorrecttPadder(self):
        padder = self.padder_factory.create(self.a_bpemb_embedding_model_type, self.a_padding_value)

        self.assertIsInstance(padder, SubwordEmbeddingPadder)
