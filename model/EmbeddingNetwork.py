import torch
import torch.nn as nn


class EmbeddingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, projection_size, batch_size, device, num_layers=1, maxpool=False, maxpool_kernel_size=3):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        
        self.model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        self.projection_layer = nn.Linear(2*hidden_size, projection_size)

        self.maxpool = maxpool
        if self.maxpool:
            self.maxpool_kernel_size = maxpool_kernel_size
            self.maxpooling_layer = nn.MaxPool1d(maxpool_kernel_size)
        else:
            self.maxpool_kernel_size = 1

    def forward(self, input_, decomposition_lengths):
        if input_.size(0) < self.batch_size:
            batch_size = 1
            self.hidden = self.__init_hidden(batch_size, self.hidden_size)
        else:
            self.hidden = self.__init_hidden(self.batch_size, self.hidden_size)

        embeddings = torch.zeros(input_.size(1), input_.size(0), int(input_.size(3) / self.maxpool_kernel_size)).cuda(self.device)

        input_ = input_.transpose(0, 1).float().cuda(self.device)

        for i in range(input_.size(0)):
            lengths = []
            for decomposition_length in decomposition_lengths:
                lengths.append(decomposition_length[i])
            packed_sequence = nn.utils.rnn.pack_padded_sequence(input_[i], torch.tensor(lengths).cuda(self.device), batch_first=True, enforce_sorted=False)

            packed_output , hidden = self.model(packed_sequence, self.hidden)
            
            padded_output, padded_output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=-32)

            word_context = torch.zeros(padded_output.size(0), padded_output.size(2)).cuda(self.device)
            for j in range(padded_output_lengths.size(0)):
                word_context[j] = padded_output[j, padded_output_lengths[j] - 1, :]

            projection_output = self.projection_layer(word_context)

            if self.maxpool:
                pooled_output = self.maxpooling_layer(projection_output.view(1, projection_output.size(0), projection_output.size(1)))
                pooled_output = pooled_output.view(pooled_output.size(1), pooled_output.size(2))

                embeddings[i] = pooled_output
            else:
                embeddings[i] = projection_output

        return embeddings.transpose(0, 1)

    def __init_hidden(self, batch_size, hidden_size):
        return (torch.zeros(1*2, batch_size, hidden_size).cuda(self.device),
                torch.zeros(1*2, batch_size, hidden_size).cuda(self.device))
