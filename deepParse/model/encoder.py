import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from deepParse.tools import weight_init


class Encoder(nn.Module):
    """
    Encoder module that use a LSTM to encode a sequence.

    Args:
        input_size (int): The input size of the decoder.
        hidden_size (int): The hidden size of the decoder.
        num_layers (int): The number of layer to the decoder.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm.apply(weight_init)

    def forward(self, to_predict, lenghts_tensor):  # todo validate input/output type
        """

        """
        packed_sequence = pack_padded_sequence(to_predict, lenghts_tensor, batch_first=True)

        _, hidden = self.lstm(packed_sequence, self.hidden)

        return hidden
