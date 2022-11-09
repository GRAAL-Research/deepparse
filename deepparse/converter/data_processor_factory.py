from regex import F
from . import DataPadder, TagsConverter, DataProcessor
from ..vectorizer import Vectorizer, BPEmbVectorizer


class DataProcessorFactory:
    def create(self, vectorizer: Vectorizer, padder: DataPadder, tags_converter: TagsConverter):
        if isinstance(vectorizer, BPEmbVectorizer):
            processor = DataProcessor(
                vectorizer, padder.pad_subword_embeddings_sequences, padder.pad_subword_embeddings_batch, tags_converter
            )

        else:
            processor = DataProcessor(
                vectorizer, padder.pad_word_embeddings_sequences, padder.pad_word_embeddings_batch, tags_converter
            )

        return processor
