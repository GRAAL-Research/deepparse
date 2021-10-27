import torch.nn as nn
import torch.nn.init as init


def weights_init(m):
    """
    Function to initialize the weights of our layers.
    Usage:
        network = Model()
        network.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
