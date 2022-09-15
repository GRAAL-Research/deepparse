from abc import ABC, abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence


class DataPadder(ABC):
    def __init__(self, padding_value) -> None:
        self.padding_value = padding_value

    def pad_batch(self, batch, teacher_forcing=False):
        sequence_batch, target_batch = self._extract_sequences_and_target(batch)

        padded_sequences_and_lengths = self.pad_sequences(sequence_batch)
        padded_target = self.pad_target(target_batch)

        if teacher_forcing:
            return padded_sequences_and_lengths + (padded_target,), padded_target

        return padded_sequences_and_lengths, padded_target

    def pad_target(self, target_batch):
        target_batch = map(torch.tensor, target_batch)

        return pad_sequence(target_batch, batch_first=True, padding_value=self.padding_value)

    @abstractmethod
    def pad_sequences(self, sequence_batch):
        pass

    @abstractmethod
    def _extract_sequences_and_target(self, batch):
        pass
