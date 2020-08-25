import torch.nn as nn

from deepParse.tools import weight_init


class Decoder(nn.Module):
    """
    Decoder module that use a LSTM to decode a previously encoded sequence and a linear layer to map
    the decoded sequence tags.

    Args:
        input_size (int): The input size of the decoder.
        hidden_size (int): The hidden size of the decoder.
        num_layers (int): The number of layer to the decoder.
        output_size (int): The output size of the decoder (i.e. the number of tags to predict on).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.lstm.apply(weight_init)

        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.apply(weight_init)

        self.softmax = nn.LogSoftmax(dim=1)

    def __call__(self, to_predict, hidden):  # todo validate input/output type
        """

        Return:
            The prediction vector.

        """
        output, _ = self.lstm(to_predict.float(), hidden)

        output_prob = self.softmax(self.linear(output[0]))

        return output_prob

    def eval(self) -> None:
        """
        To put the network in eval mode (no weights update).
        """
        self.lstm.eval()
        self.linear.eval()
