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
            batch_size = 1                  # C'est pour pouvoir gérer des batch sizes différents de façon dynamique (e.g: l'entrainement du modèle a été fait avec batch size de 2048)
            self.hidden = self.__init_hidden(batch_size, self.hidden_size) # mais il y'a des pays qui ont moins que 2048 données au total (penses au pays zero shot) 
        else:                                                               # donc si on fixe ça dans l'init ça plante
            self.hidden = self.__init_hidden(self.batch_size, self.hidden_size) # Ça va être utile pour le parser aussi car je veux garder la possibilité de traitement en batch
                                                                                # pendant l'inférence comme ça si un user a un grand nombre d'adresses on pourra optimiser
        packed_sequence = nn.utils.rnn.pack_padded_sequence(input_, lenghts_tensor, batch_first=True)

        _ , hidden = self.lstm(packed_sequence, self.hidden)

        return hidden

    def __init_hidden(self, batch_size, hidden_size):
        return (torch.zeros(1, batch_size, hidden_size).to(self.device), 
                torch.zeros(1, batch_size, hidden_size).to(self.device))


