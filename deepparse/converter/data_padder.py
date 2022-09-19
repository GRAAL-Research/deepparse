from typing import List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class DataPadder:
    """
    Class that handles the padding of vectorized sequences to the length of the longuest sequence.
    Args:
        padding_value (int): the value to use as padding to extend the shorter sequences. Default: -100.
    """

    def __init__(self, padding_value: int = -100) -> None:
        self.padding_value = padding_value

    def pad_word_embeddings_batch(
        self, batch: List[Tuple[List, List]], teacher_forcing=False
    ) -> Union[
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        sequences_vectors, target_vectors = self._extract_word_embeddings_sequences_and_target(batch)

        padded_sequences, lengths = self.pad_word_embeddings_sequences(sequences_vectors)
        padded_target_vectors = self.pad_targets(target_vectors)

        if teacher_forcing:
            return (padded_sequences, lengths, padded_target_vectors), padded_target_vectors

        return (padded_sequences, lengths), padded_target_vectors

    def pad_word_embeddings_sequences(self, sequences_batch: Tuple[List, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences_vectors, lengths = zip(
            *[
                (
                    torch.FloatTensor(np.array(seq_vectors)),
                    len(seq_vectors),
                )
                for seq_vectors in sequences_batch
            ]
        )

        lengths = torch.tensor(lengths)

        padded_sequences_vectors = self._pad_tensors(sequences_vectors)

        return padded_sequences_vectors, lengths

    def pad_subword_embeddings_batch(
        self, batch: List[Tuple[Tuple[List, List], List]], teacher_forcing=False
    ) -> Union[
        Tuple[Tuple[torch.Tensor, List, torch.Tensor], torch.Tensor],
        Tuple[Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        sequences_tuples, target_vectors = self._extract_subword_embeddings_sequences_and_targets(batch)

        padded_sequences, decomposition_lengths, sequence_lengths = self.pad_subword_embeddings_sequences(
            sequences_tuples
        )
        padded_target_vectors = self.pad_targets(target_vectors)

        if teacher_forcing:
            return (
                padded_sequences,
                decomposition_lengths,
                sequence_lengths,
                padded_target_vectors,
            ), padded_target_vectors

        return (padded_sequences, decomposition_lengths, sequence_lengths), padded_target_vectors

    def pad_subword_embeddings_sequences(
        self, sequences_batch: Tuple[Tuple[List, List], ...]
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        sequences_vectors, decomp_len, lengths = zip(
            *[
                (
                    torch.tensor(np.array(vectors)),
                    word_decomposition_len,
                    len(vectors),
                )
                for vectors, word_decomposition_len in sequences_batch
            ]
        )

        padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=self.padding_value)

        lengths = torch.tensor(lengths)
        max_sequence_length = lengths.max().item()
        for decomposition_length in decomp_len:
            if len(decomposition_length) < max_sequence_length:
                decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

        return padded_sequences_vectors, list(decomp_len), lengths

    def pad_targets(self, target_batch: Tuple[List, ...]) -> torch.Tensor:
        target_batch = map(torch.tensor, target_batch)

        return pad_sequence(target_batch, batch_first=True, padding_value=self.padding_value)

    def _extract_word_embeddings_sequences_and_target(self, batch: List[Tuple[List, List]]) -> Tuple[List, List]:
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        sequence_batch, target_batch = zip(*sorted_batch)

        return sequence_batch, target_batch

    def _extract_subword_embeddings_sequences_and_targets(
        self, batch: List[Tuple[Tuple[List, List], List]]
    ) -> Tuple[List[Tuple[List, List]], List]:
        sorted_batch = sorted(batch, key=lambda x: len(x[0][1]), reverse=True)

        sequence_batch, target_batch = zip(*sorted_batch)

        return sequence_batch, target_batch

    def _pad_tensors(self, sequences_batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        return pad_sequence(sequences_batch, batch_first=True, padding_value=self.padding_value)
