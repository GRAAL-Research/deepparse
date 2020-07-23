import pickle
import time
import os

from torch.utils.data import DataLoader
from torch import cuda, device
from torch.nn import NLLLoss
from torch.optim import SGD
from poutyne.framework import Model, Experiment
from poutyne.utils import set_seeds
import hydra
import pycountry

from data_handling.Vectorizer import Vectorizer
from data_handling.Dataset import DatasetContainer
from model.Seq2seq import Seq2seq
from data_handling.DataLoadersGenerator import DataLoadersGenerator
from utils import ToTensor
from metrics.loss import nll_loss_function
from metrics.Accuracy import accuracy


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    set_seeds(cfg.learning_hyperparameters.seed)

    tags_to_idx = {
            "StreetNumber": 0,
            "StreetName": 1,
            "Unit": 2,
            "Municipality": 3,
            "Province": 4,
            "PostalCode": 5,
            "Orientation": 6,
            "GeneralDelivery": 7
        }
    EOS_token = 8

    vectorizer = Vectorizer(cfg.learning_hyperparameters.embeddings_path, tags_to_idx, EOS_token, cfg.learning_hyperparameters.padding_value)

    train_device = device(f'cuda:{cfg.learning_hyperparameters.torch_device}' if cuda.is_available() else 'cpu')

    to_tensor = ToTensor(cfg.learning_hyperparameters.embedding_size, vectorizer, cfg.learning_hyperparameters.padding_value, train_device)

    tf_transform = to_tensor.get_teacher_forcing_from_batch()
    or_transform = to_tensor.get_output_reuse_from_batch()

    model = Seq2seq(cfg.model.embedding_input_size, cfg.model.encoder_input_size, cfg.model.decoder_input_size, cfg.model.embedding_hidden_size, cfg.model.embedding_projection_size, 
                    cfg.model.encoder_hidden_size, cfg.model.num_encoding_layers, cfg.model.decoder_hidden_size, cfg.model.num_decoding_layers, cfg.model.output_size, 
                    cfg.learning_hyperparameters.batch_size, EOS_token, train_device)

    optimizer = SGD(model.parameters(), cfg.learning_hyperparameters.learning_rate)

    loss_fn = nll_loss_function
    accuracy_fn = accuracy

    exp = Experiment(os.path.join(os.getcwd(), 'checkpoints'), model, device=train_device, optimizer=optimizer, loss_function=loss_fn, batch_metrics=[accuracy_fn])

if __name__ == "__main__":
    main()