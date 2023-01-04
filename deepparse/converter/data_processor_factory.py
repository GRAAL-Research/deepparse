from . import DataPadder, TagsConverter, DataProcessor
from ..vectorizer import Vectorizer, BPEmbVectorizer


class DataProcessorFactory:
    """
    A factory for data processors
    """

    def create(self, vectorizer: Vectorizer, padder: DataPadder, tags_converter: TagsConverter):
        """
        A factory method to create a data processor
        Args:
            vectorizer (:class:`~Vectorizer`): a callable vectorizer capable of vectorizing a list of addresses
            padder (:class:`~DataPadder`): a data padder with methods to pad address sequences and batches
            tags_converter (:class:`~TagsConverter`): a callable converter to transform address labels into indices for training
        Return:
            A :class:`~DataProcessor`
        """
        if isinstance(vectorizer, BPEmbVectorizer):
            processor = DataProcessor(
                vectorizer, padder.pad_subword_embeddings_sequences, padder.pad_subword_embeddings_batch, tags_converter
            )

        else:
            processor = DataProcessor(
                vectorizer, padder.pad_word_embeddings_sequences, padder.pad_word_embeddings_batch, tags_converter
            )

        return processor
