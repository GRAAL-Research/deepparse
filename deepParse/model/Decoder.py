import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device):
        super().__init__()

        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        output, hidden = self.lstm(input_.float(), hidden)

        output = self.softmax(self.linear(output[0]))

        return output, hidden

    def _init_hidden(self, hidden_size):
        return (torch.zeros(1, self.batch_size, hidden_size).to(self.device), 
                torch.zeros(1, self.batch_size, hidden_size).to(self.device))


