# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, too-many-arguments

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from deepparse.dataset_container import DatasetContainer, ListDatasetContainer
from deepparse.parser import CACHE_PATH, AddressParser


class AddressParserRetrainTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        # super(AddressParserRetrainTestCase, cls).setUpClass()

        a_list_dataset = [
            (
                '350 rue des Lilas Ouest Quebec city Quebec G1L 1B6',
                [
                    'StreetNumber',
                    'StreetName',
                    'StreetName',
                    'StreetName',
                    'Municipality',
                    'Municipality',
                    'Municipality',
                    'Province',
                    'PostalCode',
                    'PostalCode',
                ],
            ),
            (
                '350 rue des Lilas Ouest Quebec city Quebec G1L 1B6',
                [
                    'StreetNumber',
                    'StreetName',
                    'StreetName',
                    'StreetName',
                    'Municipality',
                    'Municipality',
                    'Municipality',
                    'Province',
                    'PostalCode',
                    'PostalCode',
                ],
            ),
        ]

        cls.training_container = ListDatasetContainer(a_list_dataset)
        cls.test_container = ListDatasetContainer(a_list_dataset)

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
