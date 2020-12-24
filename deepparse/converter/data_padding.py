# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# By default the loss and accuracy ignore the value of -100
# we leverage that when padding elements
padding_value = -100


def fasttext_data_padding(batch: List) -> Tuple:
    """
    Function that adds padding to the sequences so all can have the same length as the longest one for fastText model.

    Args:
        batch (List): The vectorized batch data.

    Returns:
        A tuple (``x``, ``y``). The element ``x`` is a tensor of padded word vectors and ``y``  is their respective
        lengths of the sequences.
    """

    sequences_vectors, lengths = zip(*[(torch.FloatTensor(seq_vectors), len(seq_vectors))
                                       for seq_vectors in sorted(batch, key=lambda x: len(x[0]), reverse=True)])

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=padding_value)

    return padded_sequences_vectors, lengths


def bpemb_data_padding(batch: List[Tuple]) -> Tuple:
    """
    Function that add padding to the sequences and to the decomposition lengths so all can have the same length as
    the longest one.

    Args:
        batch (list[tuple]): The list of vectorize tupled batch data where the first element is the address embeddings
            and the second is the word decomposition lengths.

    Returns:
        A tuple (``x``, ``y``, ``z``). The element ``x`` is a tensor of padded word vectors, ``y`` is the padded
        decomposition lengths, and ``z`` is the original lengths of the sequences before padding.
    """

    sequences_vectors, decomp_len, lengths = zip(
        *[(torch.tensor(vectors), word_decomposition_len, len(vectors))
          for vectors, word_decomposition_len in sorted(batch, key=lambda x: len(x[0]), reverse=True)])

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=padding_value)

    # pad decomposition length
    max_sequence_length = lengths.max().item()
    for decomposition_length in decomp_len:
        if len(decomposition_length) < max_sequence_length:
            decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

    return padded_sequences_vectors, list(decomp_len), lengths


def fasttext_data_padding_teacher_forcing(batch: List) -> Tuple:
    """
    Function that adds padding to the sequences so all can have the same length as the longest one,
    using teacher forcing training (i.e. we also provide the target during training).

    Args:
        batch (List): The vectorized batch data

    Returns:
        A tuple ((``x``, ``y``, ``z``), ``z``). The element ``x`` is a tensor of padded word vectors, ``y`` is their
        respective lengths of the sequences and ``z`` is a tensor of padded target idx. We use teacher forcing so we
        also need to pass the target during training (``z``).
    """

    sequences_vectors, target_vectors, lengths = _convert_sequence_to_tensor(batch)

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=padding_value)
    padded_target_vectors = pad_sequence(target_vectors, batch_first=True, padding_value=padding_value)

    return (padded_sequences_vectors, lengths, padded_target_vectors), padded_target_vectors


def bpemb_data_padding_teacher_forcing(batch: List[Tuple]) -> Tuple:
    """
    Function that add padding to the sequences and to the decomposition lengths so all can have the same length as
    the longest one, using teacher forcing training (i.e. we also provide the target during training).

    Args:
        batch (list[tuple]): The list of vectorize tupled batch data where the first element is the address embeddings
            and the second is the word decomposition lengths.

    Returns:
        A tuple ((``x``, ``y``, ``z``, ``w``), ``w``). The element ``x`` is a tensor of padded word vectors,
        ``y`` is the padded decomposition lengths, ``z`` is the original lengths of the sequences before padding, and
        ``w`` is a tensor of padded target idx. We use teacher forcing so we also need to pass the target during
        training (``w``).
    """

    sequences_vectors, decomp_len, target_vectors, lengths = _convert_bpemb_sequence_to_tensor(batch)

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=padding_value)
    padded_target_vectors = pad_sequence(target_vectors, batch_first=True, padding_value=padding_value)

    # pad decomposition length
    max_sequence_length = lengths.max().item()
    for decomposition_length in decomp_len:
        if len(decomposition_length) < max_sequence_length:
            decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

    return (padded_sequences_vectors, list(decomp_len), lengths, padded_target_vectors), padded_target_vectors


def fasttext_data_padding_with_target(batch: List) -> Tuple:
    """
    Function that adds padding to the sequences so all can have the same length as the longest one.

    Args:
        batch (List): The vectorized batch data

    Returns:
        A tuple ((``x``, ``y``), ``z``). The element ``x`` is a tensor of padded word vectors, ``y`` is their
        respective lengths of the sequences and ``z`` is a tensor of padded target idx.
    """

    sequences_vectors, target_vectors, lengths = _convert_sequence_to_tensor(batch)

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=padding_value)
    padded_target_vectors = pad_sequence(target_vectors, batch_first=True, padding_value=padding_value)

    return (padded_sequences_vectors, lengths), padded_target_vectors


def bpemb_data_padding_with_target(batch: List[Tuple]) -> Tuple:
    """
    Function that add padding to the sequences and to the decomposition lengths so all can have the same length as
    the longest one.

    Args:
        batch (list[tuple]): The list of vectorize tupled batch data where the first element is the address embeddings
            and the second is the word decomposition lengths.

    Returns:
        A tuple ((``x``, ``y`` , ``z``), ``w``). The element ``x`` is a tensor of padded word vectors,
        ``y`` is the padded decomposition lengths, ``z`` is the original lengths of the sequences before padding, and
        ``w`` is a tensor of padded target idx.
    """

    sequences_vectors, decomp_len, target_vectors, lengths = _convert_bpemb_sequence_to_tensor(batch)

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=padding_value)
    padded_target_vectors = pad_sequence(target_vectors, batch_first=True, padding_value=padding_value)

    # pad decomposition length
    max_sequence_length = lengths.max().item()
    for decomposition_length in decomp_len:
        if len(decomposition_length) < max_sequence_length:
            decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

    return (padded_sequences_vectors, list(decomp_len), lengths), padded_target_vectors


def _convert_sequence_to_tensor(batch):
    """
    Sort and convert sequence into a tensor with target element
    """
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    return zip(*[(torch.FloatTensor(seq_vectors), torch.tensor(target_vector), len(seq_vectors))
                 for seq_vectors, target_vector in sorted_batch])


def _convert_bpemb_sequence_to_tensor(batch):
    """
    Sort and convert a BPEmb sequence into a tensor with target element
    """
    sorted_batch = sorted(batch, key=lambda x: len(x[0][1]), reverse=True)
    return zip(*[(torch.tensor(vectors), word_decomposition_len, torch.tensor(target_vectors), len(vectors))
                 for (vectors, word_decomposition_len), target_vectors in sorted_batch])
