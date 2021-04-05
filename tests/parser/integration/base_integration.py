# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# no-member skip is so because child define the training_container in setup
# pylint: disable=not-callable, too-many-public-methods, no-member

import os
import shutil
from unittest import TestCase

import torch

from deepparse.parser import CACHE_PATH


class AddressParserRetrainTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
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
        cls.a_checkpoints_saving_dir = "./chekpoints"

        cls.fasttext_local_path = os.path.join(CACHE_PATH, "fasttext.ckpt")
        cls.bpemb_local_path = os.path.join(CACHE_PATH, "bpemb.ckpt")

        cls.a_torch_device = torch.device("cuda:0")

        cls.a_data_saving_dir = "./data"

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.a_data_saving_dir):
            shutil.rmtree(cls.a_data_saving_dir)

    def tearDown(self) -> None:
        self.clean_checkpoints()

    def clean_checkpoints(self):
        if os.path.exists(self.a_checkpoints_saving_dir):
            shutil.rmtree(self.a_checkpoints_saving_dir)

    def training(self, address_parser, with_new_prediction_tags=None):
        address_parser.retrain(self.training_container,
                               self.a_train_ratio,
                               epochs=self.a_single_epoch,
                               batch_size=self.a_batch_size,
                               num_workers=self.a_number_of_workers,
                               logging_path=self.a_checkpoints_saving_dir,
                               prediction_tags=with_new_prediction_tags)

    def setUp(self) -> None:
        self.clean_checkpoints()
