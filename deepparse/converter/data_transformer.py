from typing import Tuple

from deepparse.converter import fasttext_data_padding_teacher_forcing, bpemb_data_padding_teacher_forcing, \
    bpemb_data_padding_with_target, fasttext_data_padding_with_target
from deepparse.vectorizer import TrainVectorizer


class DataTransform:
    """
    Data transformer to vectorize the data and prepare it for training.

    Args:
        vectorizer (~deepparse.deepparse.train_vectorizer.TrainVectorizer): Vectorizer to vectorize the data
         (i.e. transform into word embedding and tag idx).
        model_type (str): The model type.
    """

    def __init__(self, vectorizer: TrainVectorizer, model_type: str):
        self.vectorizer = vectorizer
        if model_type in ("fasttext", "fastest"):
            self.teacher_forcing_data_padding_fn = fasttext_data_padding_teacher_forcing
            self.output_transform_data_padding_fn = fasttext_data_padding_with_target
        elif model_type in ("bpemb", "best", "lightest"):
            self.teacher_forcing_data_padding_fn = bpemb_data_padding_teacher_forcing
            self.output_transform_data_padding_fn = bpemb_data_padding_with_target

    def teacher_forcing_transform(self, batch_pairs: Tuple) -> Tuple:
        """
        Apply a teacher forcing transform (into tensor) to a batch of pairs (address, target).
        """
        vectorize_batch_pairs = self.vectorizer(batch_pairs)

        return self.teacher_forcing_data_padding_fn(vectorize_batch_pairs)

    def output_transform(self, batch_pairs: Tuple) -> Tuple:
        """
        Apply a transform (into tensor) to a batch of pairs (address, target).
        """
        vectorize_batch_pairs = self.vectorizer(batch_pairs)

        return self.output_transform_data_padding_fn(vectorize_batch_pairs)
