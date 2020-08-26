# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def data_padding(batch: List) -> Tuple:
    """
    Function that add padding to the sequences so all can have the same length as the longest one.

    Args:
        batch (List): The vectorize batch data.

    Returns:
        A tuple (x, y). The element x is a tensor of padded word vectors and their respective lengths of the sequences.
    """

    sequences_vectors, lengths = zip(*[(torch.FloatTensor(seq_vectors), len(seq_vectors))
                                       for seq_vectors in sorted(batch, reverse=True)])

    lengths = torch.LongTensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True)

    return padded_sequences_vectors, lengths


def bpemb_data_padding(batch: List[Tuple]) -> Tuple:
    """
    Function that add padding to the sequences and to the decomposition lengths so all can have the same length as
    the longest one.

    Args:
        batch (List[Tuple]): The list of vectorize tupled batch data where the first element is the address embeddings
        and the second is the word decomposition lengths.

    Returns:
        A tuple (x, y, z). The element x is a tensor of padded word vectors, y is the padded decomposition lengths,
        and z is their respective lengths of the sequences.
    """

    sequence_bpe_tensors, decomposition_lengths, lengths = zip(*[(torch.tensor(bpe_vector), word_decomposition_lengths,
                                                                  len(bpe_vector))
                                                                 for bpe_vector, word_decomposition_lengths in batch])

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequence_bpe_tensors, batch_first=True)

    # pad decomposition length
    max_sequence_length = lengths.max()
    for decomposition_length in decomposition_lengths:
        if len(decomposition_length) < max_sequence_length:
            decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

    return padded_sequences_vectors, decomposition_lengths, lengths
