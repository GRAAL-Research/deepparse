# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# no-member skip is so because child define the training_container in setup
# pylint: disable=not-callable, too-many-public-methods, no-member, too-many-arguments

import os
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

import torch

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer, DatasetContainer
from deepparse.parser import AddressParser, FormattedParsedAddress


class AddressParserBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an_address_to_parse = "350 rue des lilas o"

    def setup_model_with_config(self, config):
        self.a_model = AddressParser(**config)


class AddressParserPredictBase(AddressParserBase):
    def assert_properly_parse(self, parsed_address, multiple_address=False):
        if multiple_address:
            self.assertIsInstance(parsed_address, List)
            parsed_address = parsed_address[0]
        self.assertIsInstance(parsed_address, FormattedParsedAddress)

    def tearDown(self) -> None:
        del self.a_model


class AddressParserPredictNewParamsBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an_address_to_parse = "350 rue des lilas o"
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_data_saving_dir = os.path.join(cls.temp_dir_obj.name, "data")
        os.makedirs(cls.a_data_saving_dir, exist_ok=True)
        file_extension = "p"
        training_dataset_name = "sample_incomplete_data"
        download_from_url(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.training_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension)
        )

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
        self.a_fasttext_retrain_model_path = os.path.join(
            self.a_checkpoints_saving_dir,
            self.retrain_file_name_format.format("fasttext") + ".ckpt",
        )
        self.a_bpemb_retrain_model_path = os.path.join(
            self.a_checkpoints_saving_dir,
            self.retrain_file_name_format.format("bpemb") + ".ckpt",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def tearDown(self) -> None:
        self.training_temp_dir_obj.cleanup()

    def training(
        self,
        address_parser: AddressParser,
        data_container: DatasetContainer,
        num_workers: int,
        prediction_tags=None,
        seq2seq_params=None,
    ):
        address_parser.retrain(
            data_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=num_workers,
            logging_path=self.a_checkpoints_saving_dir,
            prediction_tags=prediction_tags,
            seq2seq_params=seq2seq_params,
        )
