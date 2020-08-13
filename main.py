import pickle
import time
import os

from torch.utils.data import DataLoader
from torch import cuda, device, load
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
from data_handling.ToTensorOuputReuse import ToTensorOuputReuse
from data_handling.ToTensorTeacerForcing import ToTensorTeacerForcing
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

    # to_tensor = ToTensor(cfg.learning_hyperparameters.embedding_size, vectorizer, cfg.learning_hyperparameters.padding_value, train_device)
    to_tensor = ToTensorOuputReuse(cfg.learning_hyperparameters.embedding_size, vectorizer, cfg.learning_hyperparameters.padding_value, train_device)

    # tf_transform = to_tensor.get_teacher_forcing_from_batch()
    or_transform = to_tensor.transform_function() #to_tensor.get_output_reuse_from_batch()

    model = Seq2seq(cfg.model.embedding_input_size, cfg.model.encoder_input_size, cfg.model.decoder_input_size, cfg.model.embedding_hidden_size, cfg.model.embedding_projection_size, 
                    cfg.model.encoder_hidden_size, cfg.model.num_encoding_layers, cfg.model.decoder_hidden_size, cfg.model.num_decoding_layers, cfg.model.output_size, 
                    cfg.learning_hyperparameters.batch_size, EOS_token, train_device)

    model.load_state_dict(load('/Users/mayas/Desktop/Projects/Publications/Leveraging subword embeddings for multinational address parsing/deepParse/checkpoint_epoch_32.ckpt', map_location=train_device))

    input_ = or_transform([('광주광역시 동구 필문대로60번길 25-5 61404', ['Province', 'Municipality', 'StreetName', 'StreetNumber', 'PostalCode'])])

    res = model(input_[0][0], input_[0][1], input_[0][2])
    print(res.max(2))
    
if __name__ == "__main__":
    main()