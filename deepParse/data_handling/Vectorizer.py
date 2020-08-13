from bpemb import BPEmb
import numpy as np

class Vectorizer:

    def __init__(self, embeddings_path, tags_to_idx, EOS_token, padding_value):
        self.bpe_embedding_model = BPEmb(lang="multi", vs=100000, dim=300)

        self.tags_to_idx = tags_to_idx

        self.EOS_token = EOS_token

        self.padding_value = padding_value

    def __call__(self, pairs_batch):
        batch = []
        max_length = 0

        for pair in pairs_batch:
            target_sequence = []
            bpe_sequence = []
            word_decomposition_lengths = []

            for word in pair[0].split():
                word_decomposition = []
                bpe_decomposition = self.bpe_embedding_model.embed(word)
                word_decomposition_lengths.append(len(bpe_decomposition))
                for i in range(bpe_decomposition.shape[0]):
                    word_decomposition.append(bpe_decomposition[i])
                bpe_sequence.append(word_decomposition)

            for decomposition in bpe_sequence:
                if len(decomposition) > max_length:
                    max_length = len(decomposition)

            for target in pair[1]:
                target_sequence.append(self.tags_to_idx[target])

            target_sequence.append(self.EOS_token)

            batch.append((bpe_sequence, word_decomposition_lengths, target_sequence))

        for decomposed_sequence, _, _ in batch:
            for decomposition in decomposed_sequence:
                if len(decomposition) != max_length:
                    for i in range(max_length - len(decomposition)):
                        decomposition.append(np.ones(300) * (self.padding_value))

        return sorted(batch, key=lambda x: len(x[0]), reverse=True)