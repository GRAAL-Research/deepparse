from typing import List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class DataPadder:
    """
    Class that handles the padding of vectorized sequences to the length of the longest sequence.
    Args:
        padding_value (int): the value to use as padding to extend the shorter sequences. Default: -100.
    """

    def __init__(self, padding_value: int = -100) -> None:
        self.padding_value = padding_value

    def pad_word_embeddings_batch(
        self, batch: List[Tuple[List, List]], teacher_forcing: bool = False
    ) -> Union[
        Tuple[Tuple[torch.Tensor, List], torch.Tensor],
        Tuple[Tuple[torch.Tensor, List, torch.Tensor], torch.Tensor],
    ]:
        """
        Method to pad a batch of word embeddings sequences and their targets to the length of the longest one.
        Args:
            batch (list[Tuple[list, list]]): a list of tuples where the first element is a list
                of word embeddings (the sequence) and the second is a list of targets.
            teacher_forcing (bool): if True, the padded target vectors are returned twice,
                once with the sequences and their lengths, and once on their own. This enables
                the use of teacher forcing during the training of sequence to sequence models.
        Return:
            A tuple of two elements:
                - a tuple containing either a
                    - (:class:`~torch.Tensor`, list) for the padded sequences and their respective original lengths, or
                    - (:class:`~torch.Tensor`, list, :class:`~torch.Tensor`) for the padded sequences, their lengths,
                        and the padded targets if `teacher_forcing` is true.
                    For details on the padding of sequences,
                    check out :meth:`~DataPadder.pad_word_embeddings_sequences` below.
                    The returned sequences are sorted in decreasing order.
                - a :class:`~torch.Tensor` containing the padded targets.
        """
        sequences_vectors, target_vectors = self._extract_word_embeddings_sequences_and_target(batch)

        padded_sequences, lengths = self.pad_word_embeddings_sequences(sequences_vectors)
        padded_target_vectors = self.pad_targets(target_vectors)

        if teacher_forcing:
            return (padded_sequences, lengths, padded_target_vectors), padded_target_vectors

        return (padded_sequences, lengths), padded_target_vectors

    def pad_word_embeddings_sequences(self, sequences_batch: List) -> Tuple[torch.Tensor, List]:
        """
        Method to pad a batch of word embeddings sequences.
        Args:
            sequences_batch (list): a tuple containing lists of word embeddings (the sequences)
        Return:
            A tuple of two elements:
                - a :class:`~torch.Tensor` containing the padded sequences.
                - a list containing the respective original lengths of the padded sequences.
        """
        sequences_vectors, lengths = zip(
            *[
                (
                    torch.FloatTensor(np.array(seq_vectors)),
                    len(seq_vectors),
                )
                for seq_vectors in sequences_batch
            ]
        )

        padded_sequences_vectors = self._pad_tensors(sequences_vectors)

        return padded_sequences_vectors, list(lengths)

    def pad_subword_embeddings_batch(
        self, batch: List[Tuple[Tuple[List, List], List]], teacher_forcing: bool = False
    ) -> Union[
        Tuple[Tuple[torch.Tensor, List, List], torch.Tensor],
        Tuple[Tuple[torch.Tensor, List, List, torch.Tensor], torch.Tensor],
    ]:
        """
        Method to pad a batch of subword embeddings sequences and their targets to the length of the longest one.
        Args:
            batch (list[Tuple[Tuple[list, list], list]]): a list of tuples containing the two following elements:
                - a tuple where the first element is a list of words represented as subword embeddings and the
                    second element is a list of the number of subword embeddings that each word is decomposed into.
                - a list of targets.
            teacher_forcing (bool): if True, the padded target vectors are returned twice,
                once with the sequences and their lengths, and once on their own. This enables
                the use of teacher forcing during the training of sequence to sequence models.
        Return:
            A tuple of two elements:
                - A tuple (``x``, ``y`` , ``z``). The element ``x`` is a :class:`~torch.Tensor` of
                    padded subword vectors,``y`` is a list of padded decomposition lengths,
                    and ``z`` is a list of the original lengths of the sequences
                    before padding. If teacher_forcing is True, a fourth element is added which
                    corresponds to a :class:`~torch.Tensor` of the padded targets. For details
                    on the padding of sequences, check out :meth:`~DataPadder.pad_subword_embeddings_sequences` below.
                    The returned sequences are sorted in decreasing order.
                - a :class:`~torch.Tensor` containing the padded targets.
        """
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
        self, sequences_batch: List[Tuple[List, List]]
    ) -> Tuple[torch.Tensor, List, List]:
        """
        Method to pad a batch of subword embeddings sequences.
        Args:
            sequences_batch (list[Tuple[list, list]]): a list of tuple containing tuples of two elements:
                - a list of lists representing words as lists of subword embeddings.
                - a list of the number of subword embeddings that each word is decomposed into.
        Return:
            A tuple of three elements:
                - a :class:`~torch.Tensor` containing the padded sequences.
                - a list containing the padded decomposition lengths of each word. When a word is
                    added as padding to elongate a sequence, we consider that the decomposition
                    length of the added word is 1.
                - a list containing the respective original lengths (number of words)
                    of the padded sequences.
        """
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

        padded_sequences_vectors = self._pad_tensors(sequences_vectors)

        longest_sequence_length = max(lengths)
        for decomposition_length in decomp_len:
            if len(decomposition_length) < longest_sequence_length:
                decomposition_length.extend([1] * (longest_sequence_length - len(decomposition_length)))

        return padded_sequences_vectors, list(decomp_len), list(lengths)

    def pad_targets(self, target_batch: List) -> torch.Tensor:
        """
        Method to pad a batch of target indices to the longest one.
        Args:
            target_batch (list): a tuple containing lists of target indices.
        Return:
            A :class:`~torch.Tensor` of padded targets.
        """
        target_batch = map(torch.tensor, target_batch)

        return self._pad_tensors(target_batch)

    def _extract_word_embeddings_sequences_and_target(self, batch: List[Tuple[List, List]]) -> Tuple[List, List]:
        """
        Method that takes a list of word embedding sequences and targets and zips the
            sequences together and the targets together.
        """
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        sequence_batch, target_batch = zip(*sorted_batch)

        return sequence_batch, target_batch

    def _extract_subword_embeddings_sequences_and_targets(
        self, batch: List[Tuple[Tuple[List, List], List]]
    ) -> Tuple[List[Tuple[List, List]], List]:
        """
        Method that takes a list of subword embedding sequences and targets
            and zips the sequences together and the targets together.
        """
        sorted_batch = sorted(batch, key=lambda x: len(x[0][1]), reverse=True)

        sequence_batch, target_batch = zip(*sorted_batch)

        return sequence_batch, target_batch

    def _pad_tensors(self, sequences_batch: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        A method to pad and collate multiple :class:``torch.Tensor` representing sequences
            into a single :class:``torch.Tensor`using :attr:`DataPadder.padding_value`.
            The final :class:``torch.Tensor` is returned with batch first
        """

        return pad_sequence(sequences_batch, batch_first=True, padding_value=self.padding_value)
