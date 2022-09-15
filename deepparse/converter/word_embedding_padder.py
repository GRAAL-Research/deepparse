import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from . import DataPadder


class WordEmbeddingPadder(DataPadder):
    def pad_sequences(self, sequence_batch):
        sequences_vectors, lengths = zip(
            *[
                (
                    torch.FloatTensor(np.array(seq_vectors)),
                    len(seq_vectors),
                )
                for seq_vectors in sequence_batch
            ]
        )

        lengths = torch.tensor(lengths)

        padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=self.padding_value)

        return padded_sequences_vectors, lengths

    def _extract_sequences_and_target(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        sequence_batch, target_batch = zip(*sorted_batch)

        return sequence_batch, target_batch
