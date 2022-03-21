# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, no-name-in-module

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from torch import tensor

from deepparse import download_from_url
from deepparse.parser import formatted_parsed_address
from tests.base_capture_output import CaptureOutputTestCase


class AddressParserPredictTestCase(CaptureOutputTestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_best_model_type = "best"
        cls.a_bpemb_model_type = "bpemb"
        cls.a_fastest_model_type = "fastest"
        cls.a_fasttext_model_type = "fasttext"
        cls.a_fasttext_lightest_model_type = "lightest"
        cls.a_fasttext_light_model_type = "fasttext-light"

        # A address parsing example
        cls.a_complete_address = "15 major st london ontario n5z1e1"
        cls.a_municipality = "london"
        cls.a_postal_code = "n5z1e1"
        cls.a_province = "ontario"
        cls.a_street_name = "major st"
        cls.a_street_number = "15"

        cls.a_logging_path = "data"

    def setUp(self):
        # a prediction vector with real values
        self.a_prediction_vector_for_a_complete_address = tensor(
            [
                [
                    [
                        -6.7080e-04,
                        -7.3572e00,
                        -1.4086e01,
                        -1.1092e01,
                        -2.1749e01,
                        -1.1060e01,
                        -1.4627e01,
                        -1.4654e01,
                        -2.8624e01,
                    ]
                ],
                [
                    [
                        -1.5119e01,
                        -1.7881e-06,
                        -1.7613e01,
                        -1.3365e01,
                        -2.9415e01,
                        -2.3198e01,
                        -2.2065e01,
                        -2.2009e01,
                        -4.0588e01,
                    ]
                ],
                [
                    [
                        -1.5922e01,
                        -1.1903e-03,
                        -1.3102e01,
                        -6.7359e00,
                        -2.4669e01,
                        -1.7328e01,
                        -1.9970e01,
                        -1.9923e01,
                        -4.0041e01,
                    ]
                ],
                [
                    [
                        -1.9461e01,
                        -1.3808e01,
                        -1.5707e01,
                        -2.0146e-05,
                        -1.0881e01,
                        -1.5345e01,
                        -2.1945e01,
                        -2.2081e01,
                        -4.6854e01,
                    ]
                ],
                [
                    [
                        -1.7136e01,
                        -1.8420e01,
                        -1.5489e01,
                        -1.5802e01,
                        -1.2159e-05,
                        -1.1350e01,
                        -2.1703e01,
                        -2.1866e01,
                        -4.2224e01,
                    ]
                ],
                [
                    [
                        -1.4736e01,
                        -1.7999e01,
                        -1.5483e01,
                        -2.1751e01,
                        -1.3005e01,
                        -3.4571e-06,
                        -1.7897e01,
                        -1.7965e01,
                        -1.4235e01,
                    ]
                ],
                [
                    [
                        -1.7509e01,
                        -1.8191e01,
                        -1.7853e01,
                        -2.6309e01,
                        -1.7179e01,
                        -1.0518e01,
                        -1.9438e01,
                        -1.9542e01,
                        -2.7060e-05,
                    ]
                ],
            ]
        )
        self.attention_mechanism_weights = self.a_prediction_vector_for_a_complete_address

        # to create the dirs for dumping the prediction tags since we mock Poutyne that usually will do it
        os.makedirs(self.a_logging_path, exist_ok=True)

        # to create the dir for the model and dump the prediction_tags.p if needed
        self.a_model_root_path = "model"
        os.makedirs(self.a_model_root_path, exist_ok=True)
        self.a_model_path = os.path.join(self.a_model_root_path, "model.p")

    def mock_predictions_vectors(self, model):
        returned_prediction_vectors = self.a_prediction_vector_for_a_complete_address
        returned_value = returned_prediction_vectors
        model.__call__().return_value = returned_value

    def mock_multiple_predictions_vectors(self, model):
        returned_prediction_vectors = torch.cat(
            (
                self.a_prediction_vector_for_a_complete_address,
                self.a_prediction_vector_for_a_complete_address,
            ),
            1,
        )
        returned_value = returned_prediction_vectors
        model.__call__().return_value = returned_value

    def setup_retrain_new_tags_model(self, address_components, model_type):
        data_dict = {
            "address_tagger_model": {"a_key": 1, "another_key": 2},
            "prediction_tags": address_components,
            "model_type": model_type,
        }
        torch.save(data_dict, self.a_model_path)

    def setup_retrain_new_params_model(self, seq2seq_params, model_type):
        data_dict = {
            "address_tagger_model": {"a_key": 1, "another_key": 2},
            "seq2seq_params": seq2seq_params,
            "model_type": model_type,
        }
        torch.save(data_dict, self.a_model_path)


class FormattedParsedAddressBase(TestCase):
    def reset_fields(self):
        # We reset the FIELDS of the address to default values since we change it in some tests
        default_value = [
            "StreetNumber",
            "Unit",
            "StreetName",
            "Orientation",
            "Municipality",
            "Province",
            "PostalCode",
            "GeneralDelivery",
            "EOS",
        ]
        self.set_fields(default_value)

    @staticmethod
    def set_fields(fields_value):
        formatted_parsed_address.FIELDS = fields_value


class PretrainedWeightsBase:
    def download_pre_trained_weights(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_path = os.path.join(self.temp_dir_obj.name, "fake_cache")

        download_from_url("retrained_fasttext_address_parser", self.fake_cache_path, "ckpt")
        self.path_to_retrain_fasttext = os.path.join(self.fake_cache_path, "retrained_fasttext_address_parser.ckpt")

        download_from_url("retrained_bpemb_address_parser", self.fake_cache_path, "ckpt")
        self.path_to_retrain_bpemb = os.path.join(self.fake_cache_path, "retrained_bpemb_address_parser.ckpt")
