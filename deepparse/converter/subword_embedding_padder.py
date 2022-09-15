import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from . import DataPadder


class SubwordEmbeddingPadder(DataPadder):
    def pad_sequences(self, sequence_batch):
        sequences_vectors, decomp_len, lengths = zip(
        *[
            (
                torch.tensor(np.array(vectors)),
                word_decomposition_len,
                len(vectors),
            )
            for vectors, word_decomposition_len in sequence_batch
        ]
    )

        padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=self.padding_value)

        lengths = torch.tensor(lengths)
        max_sequence_length = lengths.max().item()
        for decomposition_length in decomp_len:
            if len(decomposition_length) < max_sequence_length:
                decomposition_length.extend([1] * (max_sequence_length - len(decomposition_length)))

        return padded_sequences_vectors, list(decomp_len), lengths

    def _extract_sequences_and_target(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x[0][1]), reverse=True)

        sequence_batch, target_batch = zip(*sorted_batch)

        return sequence_batch, target_batch
