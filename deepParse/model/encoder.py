import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from deepParse.tools import weight_init


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm.apply(weight_init)

    def forward(self, input_, lenghts_tensor):
        packed_sequence = pack_padded_sequence(input_, lenghts_tensor, batch_first=True)

        _, hidden = self.lstm(packed_sequence, self.hidden)

        return hidden
