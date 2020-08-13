import torch
import torch.nn as nn


class ToTensor:
    def __init__(self, embedding_size, vectorizer, padding_value, device, mask_value=-100):
        self.embedding_size = embedding_size
        self.device = device
        self.vectorizer = vectorizer
        self.padding_value = padding_value
        self.mask_value = mask_value

    def _transform(self, pairs_batch):
        sequence_bpe_tensors, decomposition_lengths, target_tensors, lengths_vector = zip(*[(torch.tensor(bpe_vector), word_decomposition_lengths, torch.tensor(target_vector), len(bpe_vector)) \
        for bpe_vector, word_decomposition_lengths, target_vector in self.vectorizer(pairs_batch)])

        bpe_input_tensor = nn.utils.rnn.pad_sequence(sequence_bpe_tensors, batch_first=True, padding_value=self.padding_value)
        target_tensor = nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=self.mask_value)

        max_sequence_length = lengths_vector[0]
        for decomposition_length in decomposition_lengths:
            if len(decomposition_length) < max_sequence_length:
                for i in range(max_sequence_length - len(decomposition_length)):
                    decomposition_length.append(1)

        lengths_tensor = torch.tensor(lengths_vector)

        return bpe_input_tensor, decomposition_lengths, lengths_tensor, target_tensor

    def transform_function(self):
        raise NotImplementedError("This method is abstract, please use one of the children's implementation")