from torch import cuda, device, load
from poutyne.utils import set_seeds
import hydra

from deepParse.vectorizer.vectorizer import Vectorizer
from deepParse.model import seq2seq
from deepParse.research_code.data_handling import ToTensorOuputReuse


@hydra.main(config_path='research_code/conf/config.yaml')
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

    model = seq2seq(cfg.lstm.embedding_input_size, cfg.lstm.encoder_input_size, cfg.lstm.decoder_input_size, cfg.lstm.embedding_hidden_size, cfg.lstm.embedding_projection_size,
                    cfg.lstm.encoder_hidden_size, cfg.lstm.num_encoding_layers, cfg.lstm.decoder_hidden_size, cfg.lstm.num_decoding_layers, cfg.lstm.output_size,
                    cfg.learning_hyperparameters.batch_size, EOS_token, train_device)

    model.load_state_dict(load('/Users/mayas/Desktop/Projects/Publications/Leveraging subword embeddings for multinational address parsing/deepParse/checkpoint_epoch_32.ckpt', map_location=train_device))

    input_ = or_transform([('광주광역시 동구 필문대로60번길 25-5 61404', ['Province', 'Municipality', 'StreetName', 'StreetNumber', 'PostalCode'])])

    res = model(input_[0][0], input_[0][1], input_[0][2])
    print(res.max(2))
    
if __name__ == "__main__":
    main()