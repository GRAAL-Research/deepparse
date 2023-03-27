from typing import Callable, List, Tuple, Union

import torch

from . import TagsConverter
from ..vectorizer import Vectorizer


class DataProcessor:
    """
    Class that processes addresses into padded batches ready for training or inference
    Args:
        vectorizer (:class:`~Vectorizer`): a callable vectorizer capable of vectorizing a list of addresses
        sequences_padding_callback (Callable): a callback to pad a sequence of vectorized addresses to the
            longest, while returning the original unpadded lengths, see :class:`~deepparse.converter.Datapadder`
        batch_padding_callback (Callable): a callback to pad a sequence of vectorized addresses and their labels
            to the longest, while returning the original unpadded lengths,
            see :class:`~deepparse.converter.Datapadder`
        tags_converter (:class:`~TagsConverter`): a callable converter to transform address labels into
            indices for training

    """

    def __init__(
        self,
        vectorizer: Vectorizer,
        sequences_padding_callback: Callable,
        batch_padding_callback: Callable,
        tags_converter: TagsConverter,
    ) -> None:
        self.vectorizer = vectorizer
        self.sequences_padding_callback = sequences_padding_callback
        self.batch_padding_callback = batch_padding_callback
        self.tags_converter = tags_converter

    def process_for_inference(
        self, addresses: List[str]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List, torch.Tensor]]:
        """
        Method to vectorize addresses for inference.
        Args:
            addresses (List[str]): a list of addresses
        Return:
            Either a tuple of vectorized addresses and their respective original lengths before padding
            or a tuple of vectorized addresses their subword decomposition lengths and their respective
            original lengths before padding, depending on the vectorizing and padding methods.
        """
        return self.sequences_padding_callback(self.vectorizer(addresses))

    def process_for_training(
        self, addresses_and_targets: List[Tuple[str, List[str]]], teacher_forcing: bool = False
    ) -> Union[
        Union[
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ],
        Union[
            Tuple[Tuple[torch.Tensor, List, torch.Tensor], torch.Tensor],
            Tuple[Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor], torch.Tensor],
        ],
    ]:
        """
        Method to vectorize addresses and tags for training.
        Args:
            addresses_and_targets (List[Tuple[str, List[str]]]): a list of tuples where the first element is an
                address and the second is a list of tags.
            teacher_forcing (bool): if True, the padded target vectors are returned twice,
                once with the sequences and their lengths, and once on their own. This enables
                the use of teacher forcing during the training of sequence to sequence models.
        Return:
            A padded batch. Check out :meth:`~deepparse.converter.DataPadder.pad_word_embeddings_batch`
                and :meth:`~DataPadder.pad_subword_embeddings_batch` for more details.
        """
        input_sequence = []
        target_sequence = []

        addresses, targets = zip(*addresses_and_targets)

        input_sequence.extend(self.vectorizer(list(addresses)))

        for target_list in targets:
            target_tmp = [self.tags_converter(target) for target in target_list]
            target_tmp.append(self.tags_converter("EOS"))  # to append the End Of Sequence token
            target_sequence.append(target_tmp)

        return self.batch_padding_callback(list(zip(input_sequence, target_sequence)), teacher_forcing)
