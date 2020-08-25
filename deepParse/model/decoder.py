import torch.nn as nn

from deepParse.tools import weight_init


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.lstm.apply(weight_init)

        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.apply(weight_init)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        output, _ = self.lstm(input_.float(), hidden)

        output_prob = self.softmax(self.linear(output[0]))

        return output_prob
