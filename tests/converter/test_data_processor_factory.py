import unittest
from unittest import TestCase
from unittest.mock import Mock

from deepparse.converter import DataPadder, DataProcessorFactory
from deepparse.vectorizer import BPEmbVectorizer, FastTextVectorizer, Model2VecVectorizer


class DataProcessorFactoryTest(TestCase):
    def setUp(self):
        self.bpemb_vectorizer = BPEmbVectorizer(Mock())

        self.fasttext_vectorizer = FastTextVectorizer(Mock())

        self.padder = DataPadder()

        self.tags_converter_mock = Mock()

        self.processor_factory = DataProcessorFactory()

        self.invalid_vectorizer = "invalid vectorizer"

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

    def test_givenAModel2VecVectorizer_whenCreatingProcessor_thenShouldAssignWordEmbeddingsPaddingCallbacks(self):
        processor = self.processor_factory.create(Model2VecVectorizer(Mock()), self.padder, self.tags_converter_mock)

        self.assertEqual(processor.sequences_padding_callback, self.padder.pad_word_embeddings_sequences)
        self.assertEqual(processor.batch_padding_callback, self.padder.pad_word_embeddings_batch)

    def test_givenAnInvalidVectorizer_whenCreatingProcessor_thenShouldRaiseError(self):
        with self.assertRaises(NotImplementedError):
            self.processor_factory.create(self.invalid_vectorizer, self.padder, self.tags_converter_mock)


if __name__ == "__main__":
    unittest.main()
