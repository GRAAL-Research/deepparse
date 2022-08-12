# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, too-many-arguments

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from deepparse import download_from_public_repository
from deepparse.dataset_container import PickleDatasetContainer, DatasetContainer
from deepparse.parser import CACHE_PATH, AddressParser


class RetrainTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_data_saving_dir = os.path.join(cls.temp_dir_obj.name, "data")
        os.makedirs(cls.a_data_saving_dir, exist_ok=True)
        file_extension = "p"
        training_dataset_name = "sample_incomplete_data"
        test_dataset_name = "test_sample_data"
        download_from_public_repository(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)
        download_from_public_repository(test_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.a_train_pickle_dataset_path = os.path.join(
            cls.a_data_saving_dir, training_dataset_name + "." + file_extension
        )

        cls.a_test_pickle_dataset_path = os.path.join(cls.a_data_saving_dir, test_dataset_name + "." + file_extension)

        file_extension = "csv"
        download_from_public_repository(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.a_train_csv_dataset_path = os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()


class AddressParserRetrainTestCase(RetrainTestCase):
    @classmethod
    def setUpClass(cls):
        super(AddressParserRetrainTestCase, cls).setUpClass()

        cls.training_container = PickleDatasetContainer(cls.a_train_pickle_dataset_path)
        cls.test_container = PickleDatasetContainer(cls.a_test_pickle_dataset_path)

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
        cls.a_cpu_device = "cpu"

        cls.a_zero_number_of_workers = 0

        cls.fasttext_local_path = os.path.join(CACHE_PATH, "fasttext.ckpt")
        cls.bpemb_local_path = os.path.join(CACHE_PATH, "bpemb.ckpt")

        cls.with_new_prediction_tags = {
            "ALastTag": 0,
            "ATag": 1,
            "AnotherTag": 2,
            "EOS": 3,
        }

    def setUp(self) -> None:
        self.training_temp_dir_obj = TemporaryDirectory()
        self.a_checkpoints_saving_dir = os.path.join(self.training_temp_dir_obj.name, "checkpoints")

    def tearDown(self) -> None:
        self.training_temp_dir_obj.cleanup()

    def training(
        self,
        address_parser: AddressParser,
        train_data_container: DatasetContainer,
        val_data_container: DatasetContainer = None,
        num_workers: int = 1,
        prediction_tags=None,
    ):
        address_parser.retrain(
            train_data_container,
            val_data_container,
            train_ratio=self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=num_workers,
            logging_path=self.a_checkpoints_saving_dir,
            prediction_tags=prediction_tags,
        )
