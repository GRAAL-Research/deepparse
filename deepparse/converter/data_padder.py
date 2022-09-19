from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class DataPadder(ABC):
    """
    Class that handles the padding of vectorized sequences to the length of the longuest sequence.
    Args:
        padding_value (int): the value to use as padding to extend the shorter sequences. Default: -100.
    """

    def __init__(self, padding_value: int = -100) -> None:
        self.padding_value = padding_value

    def pad_batch(self, batch: List[Tuple[List, List]], teacher_forcing: bool = False) -> Tuple[Tuple, torch.Tensor]:
        """
        Method to pad a batch of sequences and their associated labels (ground truth/target)
        Args:
            batch (List[Tuple[List, List]]): The vectorized batch data. Each tuple in the list contains two elements,
                the first of which is the sequence and the second of which is the target
            teacher_forcing (bool): if true, the padded target is returned twice, once along with the sequences tuple
                and once on its own
        Return:
            A tuple of two elements, the first of which is the result of the :meth:`~DataPadder.pad_sequences` method and the second of which is
                the padded target.
        """
        sequence_batch, target_batch = self._extract_sequences_and_target(batch)

        padded_sequences_and_lengths = self.pad_sequences(sequence_batch)
        padded_target = self.pad_target(target_batch)

        if teacher_forcing:
            return padded_sequences_and_lengths + (padded_target,), padded_target

        return padded_sequences_and_lengths, padded_target

    def pad_target(self, target_batch: Tuple[List, ...]) -> torch.Tensor:
        target_batch = map(torch.tensor, target_batch)

        return pad_sequence(target_batch, batch_first=True, padding_value=self.padding_value)

    @abstractmethod
    def pad_sequences(self, sequence_batch):
        pass

    @abstractmethod
    def _extract_sequences_and_target(self, batch):
        pass
