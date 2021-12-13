from typing import Tuple

from . import fasttext_data_padding_teacher_forcing, bpemb_data_padding_teacher_forcing, \
    bpemb_data_padding_with_target, fasttext_data_padding_with_target
from ..vectorizer import TrainVectorizer


class DataTransform:
    """
    Data transformer to vectorize the data and prepare it for training.

    Args:
        vectorizer (~deepparse.deepparse.train_vectorizer.TrainVectorizer): Vectorizer to vectorize the data
         (i.e. transform into word embedding and tag idx).
        model_type (str): See AddressParser for model type. Only `fasttext-light` is not supported due to
            `pymagnitude-light` incompatibility.

        Note:
        Since Windows uses `spawn` instead of `fork` during multiprocess (for the data loading pre-processing
        `num_worker` > 0) we use the Gensim model, which takes more RAM (~10 GO) than the Fasttext one (~8 GO).
        It also takes a longer time to load. See here the
        `issue <https://github.com/GRAAL-Research/deepparse/issues/89>`_.
    """

    def __init__(self, vectorizer: TrainVectorizer, model_type: str):
        self.vectorizer = vectorizer
        if "fasttext" in model_type and "light" not in model_type:
            self.teacher_forcing_data_padding_fn = fasttext_data_padding_teacher_forcing
            self.output_transform_data_padding_fn = fasttext_data_padding_with_target
        elif "bpemb" in model_type:
            self.teacher_forcing_data_padding_fn = bpemb_data_padding_teacher_forcing
            self.output_transform_data_padding_fn = bpemb_data_padding_with_target
        else:
            # Note that we don't have lightest here since lightest is fasttext-light (magnitude) and we cannot train
            # with that model type (see doc note).
            raise NotImplementedError(f"There is no {model_type} network implemented. Value should be: "
                                      f"fasttext, bpemb or their attention variant.")

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
