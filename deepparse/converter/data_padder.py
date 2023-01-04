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
        self, batch: List[Tuple[List, List]], teacher_forcing: bool = False
    ) -> Union[
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        """
        Method to pad a batch of word embeddings sequences and their targets to the length of the longuest one.
        Args:
            batch (List[Tuple[List, List]]): a list of tuples where the first element is a list
                of word embeddings (the sequence) and the second is a list of targets.
            teacher_forcing (bool): if True, the padded target vectors are returned twice,
                once with the sequences and their lengths, and once on their own. This enables
                the use of teacher forcing during the training of sequence to sequence models.
        Return:
            A tuple of two elements:
                - a tuple containing either two :class:`~torch.Tensor` (the padded sequences and their
                    repective original lengths),or three :class:`~torch.Tensor` (the padded sequences
                    and their lengths, as well as the padded targets) if `teacher_forcing` is true.
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

    def pad_word_embeddings_sequences(self, sequences_batch: Tuple[List, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to pad a batch of word embeddings sequences.
        Args:
            seuqnces_batch (Tuple[List, ...]): a tuple containing lists of word embeddings (the sequences)
        Return:
            A tuple of two elements:
                - a :class:`~torch.Tensor` containing the padded sequcences.
                - a :class:`~torch.Tensor` containing the respective original lengths of the padded sequences.
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

        lengths = torch.tensor(lengths)

        padded_sequences_vectors = self._pad_tensors(sequences_vectors)

        return padded_sequences_vectors, lengths

    def pad_subword_embeddings_batch(
        self, batch: List[Tuple[Tuple[List, List], List]], teacher_forcing: bool = False
    ) -> Union[
        Tuple[Tuple[torch.Tensor, List, torch.Tensor], torch.Tensor],
        Tuple[Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        """
        Method to pad a batch of subword embeddings sequences and their targets to the length of the longuest one.
        Args:
            batch (List[Tuple[Tuple[List, List], List]]): a list of tuples containing the two following elements:
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
                    and ``z`` is a :class:`~torch.Tensor` of the original lengths of the sequences
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
        self, sequences_batch: Tuple[Tuple[List, List], ...]
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """
        Method to pad a batch of subword embeddings sequences.
        Args:
            sequences_batch (Tuple[Tuple[List, List], ...]): a tuple containing tuples of two elements:
                - a list of lists representing words as lists of subword embeddings.
                - a list of the number of subword embeddings that each word is decomposed into.
        Return:
            A tuple of three elements:
                - a :class:`~torch.Tensor` containing the padded sequcences.
                - a list containing the padded decomposition lengths of each word. When a word is
                    added as padding to elongate a sequence, we consider that the decomposition
                    length of the added word is 1.
                - a :class:`~torch.Tensor` containing the respective original lengths (number of words)
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

        lengths = torch.tensor(lengths)
        max_sequence_length = lengths.max().item()
        for decomposition_length in decomp_len:
            if len(decomposition_length) < max_sequence_length:
                decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

        return padded_sequences_vectors, list(decomp_len), lengths

    def pad_targets(self, target_batch: Tuple[List, ...]) -> torch.Tensor:
        """
        Method to pad a batch of target indices to the longuest one.
        Args:
            target_batch (Tuple[List, ...]): a tuple comtaining lists of target indices.
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

    def _pad_tensors(self, sequences_batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        A method to pad and collate multiple :class:``torch.Tensor` representing sequences
            into a single :class:``torch.Tensor`using :attr:`DataPadder.padding_value`.
            The final :class:``torch.Tensor` is returned with batch first
        """

        return pad_sequence(sequences_batch, batch_first=True, padding_value=self.padding_value)
