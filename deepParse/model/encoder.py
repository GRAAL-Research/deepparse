import torch
import torch.nn as nn

from deepParse.tools import weight_init


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm.apply(weight_init)

    def forward(self, input_, lenghts_tensor):
        if input_.size(0) < self.batch_size: # @Marouane à quoi cela sert-il ? On ne peut pas jsute le faire dans l'init ?
            batch_size = 1
            self.hidden = self.__init_hidden(batch_size, self.hidden_size)
        else:
            self.hidden = self.__init_hidden(self.batch_size, self.hidden_size)

        packed_sequence = nn.utils.rnn.pack_padded_sequence(input_, lenghts_tensor, batch_first=True)

        _ , hidden = self.lstm(packed_sequence, self.hidden)

        return hidden

    def __init_hidden(self, batch_size, hidden_size):
        return (torch.zeros(1, batch_size, hidden_size).to(self.device), 
                torch.zeros(1, batch_size, hidden_size).to(self.device))


