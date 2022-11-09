import unittest
from unittest import TestCase
from unittest.mock import Mock

from deepparse.converter import DataProcessorFactory, DataPadder
from deepparse.vectorizer import BPEmbVectorizer, FastTextVectorizer


class DataProcessorFactoryTest(TestCase):
    def setUp(self):
        self.bpemb_vectorizer = BPEmbVectorizer(Mock())

        self.fasttext_vectorizer = FastTextVectorizer(Mock())

        self.padder = DataPadder()

        self.tags_converter_mock = Mock()

        self.processor_factory = DataProcessorFactory()

    def test_givenABpembVectorizer_whenCreatingProcessor_thenShouldAssignCorrectSequencesPaddingCallbacks(self):
        processor = self.processor_factory.create(self.bpemb_vectorizer, self.padder, self.tags_converter_mock)

        self.assertTrue(
            processor.sequences_padding_callback.__qualname__
            == DataPadder.pad_subword_embeddings_sequences.__qualname__
        )

    def test_givenABpembVectorizer_whenCreatingProcessor_thenShouldAssignCorrectBatchPaddingCallbacks(self):
        processor = self.processor_factory.create(self.bpemb_vectorizer, self.padder, self.tags_converter_mock)

        self.assertTrue(
            processor.batch_padding_callback.__qualname__ == DataPadder.pad_subword_embeddings_batch.__qualname__
        )

    def test_givenANonBpembVectorizer_whenCreatingProcessor_thenShouldAssignCorrectSequencesPaddingCallbacks(self):
        processor = self.processor_factory.create(self.fasttext_vectorizer, self.padder, self.tags_converter_mock)

        self.assertTrue(
            processor.sequences_padding_callback.__qualname__ == DataPadder.pad_word_embeddings_sequences.__qualname__
        )

    def test_givenANonBpembVectorizer_whenCreatingProcessor_thenShouldAssignCorrectBatchPaddingCallbacks(self):
        processor = self.processor_factory.create(self.fasttext_vectorizer, self.padder, self.tags_converter_mock)

        self.assertTrue(
            processor.batch_padding_callback.__qualname__ == DataPadder.pad_word_embeddings_batch.__qualname__
        )
