# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods, no-name-in-module

import os
from unittest.mock import Mock

import torch
from torch import tensor

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
            [[[
                -6.7080e-04, -7.3572e+00, -1.4086e+01, -1.1092e+01, -2.1749e+01, -1.1060e+01, -1.4627e+01, -1.4654e+01,
                -2.8624e+01
            ]],
             [[
                 -1.5119e+01, -1.7881e-06, -1.7613e+01, -1.3365e+01, -2.9415e+01, -2.3198e+01, -2.2065e+01, -2.2009e+01,
                 -4.0588e+01
             ]],
             [[
                 -1.5922e+01, -1.1903e-03, -1.3102e+01, -6.7359e+00, -2.4669e+01, -1.7328e+01, -1.9970e+01, -1.9923e+01,
                 -4.0041e+01
             ]],
             [[
                 -1.9461e+01, -1.3808e+01, -1.5707e+01, -2.0146e-05, -1.0881e+01, -1.5345e+01, -2.1945e+01, -2.2081e+01,
                 -4.6854e+01
             ]],
             [[
                 -1.7136e+01, -1.8420e+01, -1.5489e+01, -1.5802e+01, -1.2159e-05, -1.1350e+01, -2.1703e+01, -2.1866e+01,
                 -4.2224e+01
             ]],
             [[
                 -1.4736e+01, -1.7999e+01, -1.5483e+01, -2.1751e+01, -1.3005e+01, -3.4571e-06, -1.7897e+01, -1.7965e+01,
                 -1.4235e+01
             ]],
             [[
                 -1.7509e+01, -1.8191e+01, -1.7853e+01, -2.6309e+01, -1.7179e+01, -1.0518e+01, -1.9438e+01, -1.9542e+01,
                 -2.7060e-05
             ]]])

        # to create the dirs for dumping the prediction tags since we mock Poutyne that usually will do it
        os.makedirs(self.a_logging_path, exist_ok=True)

        # to create the dir for the model and dump the prediction_tags.p if needed
        self.a_model_root_path = "model"
        os.makedirs(self.a_model_root_path, exist_ok=True)
        self.a_model_path = os.path.join(self.a_model_root_path, "model.p")

    def mock_predictions_vectors(self, model):
        model.return_value = Mock(return_value=self.a_prediction_vector_for_a_complete_address)

    def mock_multiple_predictions_vectors(self, model):
        model.return_value = Mock(return_value=torch.cat((self.a_prediction_vector_for_a_complete_address,
                                                          self.a_prediction_vector_for_a_complete_address), 1))

    def setup_retrain_new_tags_model(self, address_components, model_type):
        data_dict = {
            "address_tagger_model": {
                "a_key": 1,
                "another_key": 2
            },
            "prediction_tags": address_components,
            "model_type": model_type
        }
        torch.save(data_dict, self.a_model_path)
