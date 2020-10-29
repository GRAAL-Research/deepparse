import torch

from deepparse.converter import data_padding_teacher_forcing, bpemb_data_padding_teacher_forcing


class ToTensor:
    def __init__(self, vectorizer, device):
        self.device = device
        self.vectorizer = vectorizer

    def _teacher_forcing_transform(self, pairs):
        vectorize_pairs = self.vectorizer(pairs)

        return bpemb_data_padding_teacher_forcing(vectorize_pairs)

    def _output_reuse_transform(self, pair):
        pair = self.vectorizer(pair[0])
        input_tensor = torch.tensor(pair[0]).view(1, len(pair[0]), self.embedding_size).cuda(self.device)
        target_tensor = torch.tensor(pair[1]).view(1, -1).cuda(self.device)

        return ((input_tensor,), target_tensor)

    def get_teacher_forcing_from_batch(self):
        return self._teacher_forcing_transform

    def get_output_reuse_from_batch(self):
        return self._output_reuse_transform
