# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# no-member skip is so because child define the training_container in setup
# pylint: disable=not-callable, too-many-public-methods, no-member

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer, DatasetContainer
from deepparse.parser import CACHE_PATH, AddressParser


class AddressParserRetrainTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_temp_dir_obj = TemporaryDirectory()
        cls.a_data_saving_dir = os.path.join(cls.data_temp_dir_obj.name, "data")
        os.makedirs(cls.a_data_saving_dir, exist_ok=True)
        file_extension = "p"
        training_dataset_name = "sample_incomplete_data"
        test_dataset_name = "test_sample_data"
        download_from_url(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)
        download_from_url(test_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.training_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension))
        cls.test_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, test_dataset_name + "." + file_extension))

        cls.a_fasttext_model_type = "fasttext"
        cls.a_fasttext_light_model_type = "fasttext-light"
        cls.a_bpemb_model_type = "bpemb"

        cls.verbose = False

        # training constant
        cls.a_single_epoch = 1
        cls.a_three_epoch = 3
        cls.a_train_ratio = 0.8
        cls.a_batch_size = 128
        cls.a_number_of_workers = 2
        cls.a_learning_rate = 0.001

        cls.a_torch_device = torch.device("cuda:0")
        cls.a_cpu_device = 'cpu'

        cls.a_zero_number_of_workers = 0

        cls.fasttext_local_path = os.path.join(CACHE_PATH, "fasttext.ckpt")
        cls.bpemb_local_path = os.path.join(CACHE_PATH, "bpemb.ckpt")

        cls.with_new_prediction_tags = {'ALastTag': 0, 'ATag': 1, 'AnotherTag': 2, "EOS": 3}

    def setUp(self) -> None:
        self.training_temp_dir_obj = TemporaryDirectory()
        self.a_checkpoints_saving_dir = os.path.join(self.training_temp_dir_obj.name, "checkpoints")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.data_temp_dir_obj.cleanup()

    def tearDown(self) -> None:
        self.training_temp_dir_obj.cleanup()

    def training(self,
                 address_parser: AddressParser,
                 data_container: DatasetContainer,
                 num_workers: int,
                 prediction_tags=None):
        address_parser.retrain(data_container,
                               self.a_train_ratio,
                               epochs=self.a_single_epoch,
                               batch_size=self.a_batch_size,
                               num_workers=num_workers,
                               logging_path=self.a_checkpoints_saving_dir,
                               prediction_tags=prediction_tags)
