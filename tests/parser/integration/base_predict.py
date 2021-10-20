# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# no-member skip is so because child define the training_container in setup
# pylint: disable=not-callable, too-many-public-methods, no-member, too-many-arguments

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer, DatasetContainer
from deepparse.parser import AddressParser


class AddressParserPredictBase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fasttext_address_parser = AddressParser(model_type="fasttext", device="cpu", verbose=False)
        cls.bpemb_address_parser = AddressParser(model_type="bpemb", device="cpu", verbose=False)
        cls.an_address_to_parse = "350 rue des lilas o"


class AddressParserPredictBaseGPU(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fasttext_address_parser = AddressParser(model_type="fasttext", device=torch.device("cuda:0"), verbose=False)
        cls.bpemb_address_parser = AddressParser(model_type="bpemb", device=torch.device("cuda:0"), verbose=False)
        cls.an_address_to_parse = "350 rue des lilas o"


class AddressParserPredictNewParamsBase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.an_address_to_parse = "350 rue des lilas o"
        cls.data_temp_dir_obj = TemporaryDirectory()
        cls.a_data_saving_dir = os.path.join(cls.data_temp_dir_obj.name, "data")
        os.makedirs(cls.a_data_saving_dir, exist_ok=True)
        file_extension = "p"
        training_dataset_name = "sample_incomplete_data"
        download_from_url(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.training_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension))

        cls.a_fasttext_model_type = "fasttext"
        cls.a_bpemb_model_type = "bpemb"

        cls.verbose = False

        # training constant
        cls.a_single_epoch = 1
        cls.a_train_ratio = 0.8
        cls.a_batch_size = 128
        cls.a_number_of_workers = 2
        cls.a_learning_rate = 0.001

        cls.a_torch_device = torch.device("cuda:0")
        cls.a_cpu_device = "cpu"

        cls.seq2seq_params = {"encoder_hidden_size": 512, "decoder_hidden_size": 512}

        cls.retrain_file_name_format = "retrained_{}_address_parser"

    def setUp(self) -> None:
        self.training_temp_dir_obj = TemporaryDirectory()
        self.a_checkpoints_saving_dir = os.path.join(self.training_temp_dir_obj.name, "checkpoints")
        self.a_fasttext_retrain_model_path = os.path.join(self.a_checkpoints_saving_dir,
                                                          self.retrain_file_name_format.format("fasttext") + ".ckpt")
        self.a_bpemb_retrain_model_path = os.path.join(self.a_checkpoints_saving_dir,
                                                       self.retrain_file_name_format.format("bpemb") + ".ckpt")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.data_temp_dir_obj.cleanup()

    def tearDown(self) -> None:
        self.training_temp_dir_obj.cleanup()

    def training(self,
                 address_parser: AddressParser,
                 data_container: DatasetContainer,
                 num_workers: int,
                 prediction_tags=None,
                 seq2seq_params=None):
        address_parser.retrain(data_container,
                               self.a_train_ratio,
                               epochs=self.a_single_epoch,
                               batch_size=self.a_batch_size,
                               num_workers=num_workers,
                               logging_path=self.a_checkpoints_saving_dir,
                               prediction_tags=prediction_tags,
                               seq2seq_params=seq2seq_params)
