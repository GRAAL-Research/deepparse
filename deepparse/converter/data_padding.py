# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def data_padding(batch: List) -> Tuple:
    """
    Function that adds padding to the sequences so all can have the same length as the longest one.

    Args:
        batch (List): The vectorized batch data.

    Returns:
        A tuple (``x`` , ``y``). The element ``x``  is a tensor of padded word vectors and ``y``  is their respective
        lengths of the sequences.
    """

    sequences_vectors, lengths = zip(*[(torch.FloatTensor(seq_vectors), len(seq_vectors))
                                       for seq_vectors in sorted(batch, key=lambda x: len(x[0]), reverse=True)])

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True)

    return padded_sequences_vectors, lengths


def bpemb_data_padding(batch: List[Tuple]) -> Tuple:
    """
    Function that add padding to the sequences and to the decomposition lengths so all can have the same length as
    the longest one.

    Args:
        batch (list[tuple]): The list of vectorize tupled batch data where the first element is the address embeddings
            and the second is the word decomposition lengths.

    Returns:
        A tuple (``x`` , ``y`` , ``z``). The element ``x``  is a tensor of padded word vectors, ``y``  is the padded
        decomposition lengths, and ``z``  is the original lengths of the sequences before padding.
    """

    sequences_vectors, decomp_len, lengths = zip(
        *[(torch.tensor(vectors), word_decomposition_len, len(vectors))
          for vectors, word_decomposition_len in sorted(batch, key=lambda x: len(x[0]), reverse=True)])

    lengths = torch.tensor(lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True)

    # pad decomposition length
    max_sequence_length = lengths.max().item()
    for decomposition_length in decomp_len:
        if len(decomposition_length) < max_sequence_length:
            decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

    return padded_sequences_vectors, list(decomp_len), lengths
