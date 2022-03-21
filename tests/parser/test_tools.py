# pylint: disable=too-many-public-methods
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import skipIf

import torch

from deepparse.parser.tools import (
    indices_splitting,
    load_tuple_to_device,
    validate_if_new_prediction_tags,
    validate_if_new_seq2seq_params,
    get_address_parser_in_directory,
    get_files_in_directory,
    pretrained_parser_in_directory,
    handle_model_name,
    infer_model_type,
)
from tests.base_capture_output import CaptureOutputTestCase
from tests.parser.base import PretrainedWeightsBase
from tests.tools import create_file


class ToolsTests(CaptureOutputTestCase, PretrainedWeightsBase):
    @classmethod
    def setUpClass(cls):
        cls.download_pre_trained_weights(cls)

    def setUp(self) -> None:
        self.a_seed = 42
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_directory = self.temp_dir_obj.name

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def populate_directory(self, with_retrain_parser: bool = False):
        os.makedirs(os.path.join(self.fake_directory, "a_directory"), exist_ok=True)
        create_file(os.path.join(self.fake_directory, "afile.txt"), "a content")
        create_file(os.path.join(self.fake_directory, "another_file.txt"), "a content")
        create_file(os.path.join(self.fake_directory, "random_file.txt"), "a content")

        checkpoints_dir_path = os.path.join(self.fake_directory, "checkpoints_dir")
        os.makedirs(checkpoints_dir_path, exist_ok=True)
        create_file(os.path.join(checkpoints_dir_path, "random_file.txt"), "a content")
        if with_retrain_parser:
            create_file(
                os.path.join(checkpoints_dir_path, "retrained_fasttext_address_parser.ckpt"),
                "a content",
            )

    def test_givenACheckpointNewTags_whenValidateIfNewTags_thenReturnTrue(self):
        a_checkpoint_weights = {
            "some_weights": [1, 2, 2],
            "prediction_tags": {"a_tag": 1},
        }
        actual = validate_if_new_prediction_tags(a_checkpoint_weights)

        self.assertTrue(actual)

    def test_givenACheckpointNoNewTags_whenValidateIfNewTags_thenReturnFalse(self):
        a_checkpoint_weights = {"some_weights": [1, 2, 2]}
        actual = validate_if_new_prediction_tags(a_checkpoint_weights)

        self.assertFalse(actual)

    def test_givenACheckpointNewParams_whenValidateIfParams_thenReturnTrue(self):
        a_checkpoint_weights = {
            "some_weights": [1, 2, 2],
            "seq2seq_params": {"params": 1},
        }
        actual = validate_if_new_seq2seq_params(a_checkpoint_weights)

        self.assertTrue(actual)

    def test_givenACheckpointNoNewParams_whenValidateIfParams_thenReturnFalse(self):
        a_checkpoint_weights = {"some_weights": [1, 2, 2]}
        actual = validate_if_new_seq2seq_params(a_checkpoint_weights)

        self.assertFalse(actual)

    def test_givenADirectoryWithARetrainedModel_whenPretrainedParserInDirectory_thenReturnTrue(
        self,
    ):
        self.populate_directory(with_retrain_parser=True)
        actual = pretrained_parser_in_directory(self.fake_directory)

        self.assertTrue(actual)

    def test_givenADirectoryWithoutARetrainedModel_whenPretrainedParserInDirectory_thenReturnFalse(
        self,
    ):
        self.populate_directory(with_retrain_parser=False)
        actual = pretrained_parser_in_directory(self.fake_directory)

        self.assertFalse(actual)

    def test_givenADirectory_whenGetFilesInDirectory_thenReturnListWithAllFiles(self):
        self.populate_directory()
        actual = get_files_in_directory(self.fake_directory)

        expected = [
            "afile.txt",
            "random_file.txt",
            "another_file.txt",
            "random_file.txt",
        ]

        for actual_element in actual:
            self.assertIn(actual_element, expected)
        self.assertEqual(len(actual), len(expected))

    def test_givenAEmptyDirectory_whenGetFilesInDirectory_thenReturnEmptyList(self):
        actual = get_files_in_directory(self.fake_directory)

        expected = []

        self.assertEqual(actual, expected)

    def test_givenAEmptyDirectory_whenGetAddressParserInDirectory_thenReturnEmptyList(
        self,
    ):
        a_list_of_directory = ["afile.txt", "another_file.txt"]
        actual = get_address_parser_in_directory(a_list_of_directory)

        expected = []

        self.assertEqual(actual, expected)

    def test_givenADirectoryWithARetrainParser_whenGetAddressParserInDirectory_thenReturnThePath(
        self,
    ):
        a_list_of_directory = [
            "afile.txt",
            "another_file.txt",
            "retrained_fasttext_address_parser.ckpt",
        ]
        actual = get_address_parser_in_directory(a_list_of_directory)

        expected = ["retrained_fasttext_address_parser.ckpt"]

        self.assertEqual(actual, expected)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenATupleToLoadOfTensorIntoDevice_whenLoad_thenProperlyLoad(self):
        a_device = torch.device("cuda:0")
        a_random_tensor = torch.rand(1, 2)
        a_tuple_of_tensor = (a_random_tensor, a_random_tensor, a_random_tensor)

        actual = load_tuple_to_device(a_tuple_of_tensor, a_device)

        for element in actual:
            self.assertEqual(element.device, a_device)

    def test_givenADataset_whenIndicesSplittingRatio8020_thenSplitIndices80Train20Valid(
        self,
    ):
        number_of_data_points_in_dataset = 100
        train_ratio = 0.8
        expected_train_indices = [
            83,
            53,
            70,
            45,
            44,
            39,
            22,
            80,
            10,
            0,
            18,
            30,
            73,
            33,
            90,
            4,
            76,
            77,
            12,
            31,
            55,
            88,
            26,
            42,
            69,
            15,
            40,
            96,
            9,
            72,
            11,
            47,
            85,
            28,
            93,
            5,
            66,
            65,
            35,
            16,
            49,
            34,
            7,
            95,
            27,
            19,
            81,
            25,
            62,
            13,
            24,
            3,
            17,
            38,
            8,
            78,
            6,
            64,
            36,
            89,
            56,
            99,
            54,
            43,
            50,
            67,
            46,
            68,
            61,
            97,
            79,
            41,
            58,
            48,
            98,
            57,
            75,
            32,
            94,
            59,
        ]
        expected_valid_indices = [
            63,
            84,
            37,
            29,
            1,
            52,
            21,
            2,
            23,
            87,
            91,
            74,
            86,
            82,
            20,
            60,
            71,
            14,
            92,
            51,
        ]
        expected_len_train_indices = 80
        expected_len_valid_indices = 20

        actual_train_indices, actual_valid_indices = indices_splitting(
            number_of_data_points_in_dataset, train_ratio, seed=self.a_seed
        )
        self.assertEqual(len(actual_train_indices), expected_len_train_indices)
        self.assertEqual(len(actual_valid_indices), expected_len_valid_indices)
        self.assertEqual(actual_train_indices, expected_train_indices)
        self.assertEqual(actual_valid_indices, expected_valid_indices)

    def test_givenModelTypes_whenHandleThem_then_ReturnProperModelType(self):
        # "Normal" Fasttext setup
        model_types = ["fasttext", "fastest"]
        attention_mechanism_settings = [True, False]

        for model_type in model_types:
            for attention_mechanism_setting in attention_mechanism_settings:
                expected_model_type = "fasttext"
                actual_model_type, _ = handle_model_name(model_type, attention_mechanism=attention_mechanism_setting)
                if attention_mechanism_setting:
                    expected_model_type += "Attention"
                self.assertEqual(expected_model_type, actual_model_type)

        # fasttext-light setup
        expected_model_type = "fasttext-light"
        model_types = ["fasttext-light", "lightest"]
        for model_type in model_types:
            actual_model_type, _ = handle_model_name(model_type, attention_mechanism=False)
            self.assertEqual(expected_model_type, actual_model_type)

        # BPEmb setup
        model_types = ["bpemb", "best"]
        attention_mechanism_settings = [True, False]

        for model_type in model_types:
            for attention_mechanism_setting in attention_mechanism_settings:
                expected_model_type = "bpemb"
                actual_model_type, _ = handle_model_name(model_type, attention_mechanism=attention_mechanism_setting)
                if attention_mechanism_setting:
                    expected_model_type += "Attention"
                self.assertEqual(expected_model_type, actual_model_type)

    def test_givenModelTypes_whenHandleThem_then_ReturnProperFormattedModelType(self):
        # "Normal" Fasttext setup
        model_types = ["fasttext", "fastest"]
        attention_mechanism_settings = [True, False]

        for model_type in model_types:
            for attention_mechanism_setting in attention_mechanism_settings:
                expected_formatted_model_type = "FastText"
                _, actual_formatted_model_type = handle_model_name(
                    model_type, attention_mechanism=attention_mechanism_setting
                )
                if attention_mechanism_setting:
                    expected_formatted_model_type += "Attention"
                self.assertEqual(expected_formatted_model_type, actual_formatted_model_type)

        # fasttext-light setup
        expected_formatted_model_type = "FastTextLight"
        model_types = ["fasttext-light", "lightest"]
        for model_type in model_types:
            _, actual_formatted_model_type = handle_model_name(model_type, attention_mechanism=False)
            self.assertEqual(expected_formatted_model_type, actual_formatted_model_type)

        # BPEmb setup
        model_types = ["bpemb", "best"]
        attention_mechanism_settings = [True, False]

        for model_type in model_types:
            for attention_mechanism_setting in attention_mechanism_settings:
                expected_formatted_model_type = "BPEmb"
                _, actual_formatted_model_type = handle_model_name(
                    model_type, attention_mechanism=attention_mechanism_setting
                )
                if attention_mechanism_setting:
                    expected_formatted_model_type += "Attention"
                self.assertEqual(expected_formatted_model_type, actual_formatted_model_type)

    def test_givenAModelTypeWithAttentionInName_whenHandleModelNameWithAttFlag_thenReturnProperModelType(self):
        expected_model_type = "fasttextAttention"
        actual_model_type, _ = handle_model_name("fasttextAttention", attention_mechanism=True)
        self.assertEqual(expected_model_type, actual_model_type)

        expected_model_type = "bpembAttention"
        actual_model_type, _ = handle_model_name("bpembAttention", attention_mechanism=True)
        self.assertEqual(expected_model_type, actual_model_type)

    def test_givenAModelTypeWithAttentionInName_whenHandleModelNameWithAttFlag_thenReturnProperFormattedModelType(self):
        expected_formatted_model_type = "FastTextAttention"
        _, actual_formatted_model_type = handle_model_name("fasttextAttention", attention_mechanism=True)
        self.assertEqual(expected_formatted_model_type, actual_formatted_model_type)

        expected_formatted_model_type = "BPEmbAttention"
        _, actual_formatted_model_type = handle_model_name("bpembAttention", attention_mechanism=True)
        self.assertEqual(expected_formatted_model_type, actual_formatted_model_type)

    def test_givenAModelTypeWithAttentionInName_whenHandleModelNameWithoutAttFlag_thenRaiseError(self):
        with self.assertRaises(ValueError):
            handle_model_name("fasttextAttention", attention_mechanism=False)

        with self.assertRaises(ValueError):
            handle_model_name("bpembAttention", attention_mechanism=False)

    def test_givenAInvalidModelType_whenHandleModelName_thenRaiseError(self):
        model_type = "invalid_model_type"
        with self.assertRaises(ValueError):
            handle_model_name(model_type, attention_mechanism=False)

        expect_error_message = (
            f"Could not handle {model_type}. Read the docs at https://deepparse.org/ for possible model types."
        )

        try:
            handle_model_name(model_type, attention_mechanism=False)
        except ValueError as actual_error_message:
            self.assertEqual(actual_error_message.args[0], expect_error_message)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_givenAModelTypeToInfer_whenNotRealRetrainFastText_thenReturnFasttext(self):
        path_to_retrained_model = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "fasttext.ckpt")
        checkpoint_weights = torch.load(path_to_retrained_model, map_location="cpu")
        attention_mechanism = False

        expected_inferred_model_type = "fasttext"
        expected_attention_mechanism = False

        actual_inferred_model_type, actual_inferred_attention_mechanism = infer_model_type(
            checkpoint_weights, attention_mechanism
        )

        self.assertEqual(expected_inferred_model_type, actual_inferred_model_type)
        self.assertEqual(expected_attention_mechanism, actual_inferred_attention_mechanism)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_givenAModelTypeToInfer_whenNotRealRetrainFastTextAttention_thenReturnAttention(self):
        path_to_retrained_model = os.path.join(
            os.path.expanduser("~"), ".cache", "deepparse", "fasttext_attention.ckpt"
        )
        checkpoint_weights = torch.load(path_to_retrained_model, map_location="cpu")
        attention_mechanism = False

        expected_inferred_model_type = "fasttext"
        expected_attention_mechanism = True

        actual_inferred_model_type, actual_inferred_attention_mechanism = infer_model_type(
            checkpoint_weights, attention_mechanism
        )

        self.assertEqual(expected_inferred_model_type, actual_inferred_model_type)
        self.assertEqual(expected_attention_mechanism, actual_inferred_attention_mechanism)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_givenAModelTypeToInfer_whenNotRealRetrainBPEmb_thenReturnBPEmb(self):
        path_to_retrained_model = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "bpemb.ckpt")
        checkpoint_weights = torch.load(path_to_retrained_model, map_location="cpu")
        attention_mechanism = False

        expected_inferred_model_type = "bpemb"
        expected_attention_mechanism = False

        actual_inferred_model_type, actual_inferred_attention_mechanism = infer_model_type(
            checkpoint_weights, attention_mechanism
        )

        self.assertEqual(expected_inferred_model_type, actual_inferred_model_type)
        self.assertEqual(expected_attention_mechanism, actual_inferred_attention_mechanism)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_givenAModelTypeToInfer_whenNotRealRetrainBPEmbAttention_thenReturnAttention(self):
        path_to_retrained_model = os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "bpemb_attention.ckpt")
        checkpoint_weights = torch.load(path_to_retrained_model, map_location="cpu")
        attention_mechanism = False

        expected_inferred_model_type = "bpemb"
        expected_attention_mechanism = True

        actual_inferred_model_type, actual_inferred_attention_mechanism = infer_model_type(
            checkpoint_weights, attention_mechanism
        )

        self.assertEqual(expected_inferred_model_type, actual_inferred_model_type)
        self.assertEqual(expected_attention_mechanism, actual_inferred_attention_mechanism)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_givenAModelTypeToInfer_whenRealRetrainFastText_thenReturnFastText(self):
        path_to_retrained_model = self.path_to_retrain_fasttext
        checkpoint_weights = torch.load(path_to_retrained_model, map_location="cpu")
        attention_mechanism = False

        expected_inferred_model_type = "fasttext"

        actual_inferred_model_type, _ = infer_model_type(checkpoint_weights, attention_mechanism)

        self.assertEqual(expected_inferred_model_type, actual_inferred_model_type)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_givenAModelTypeToInfer_whenRealRetrainBPEmb_thenReturnBPEmb(self):
        path_to_retrained_model = self.path_to_retrain_bpemb
        checkpoint_weights = torch.load(path_to_retrained_model, map_location="cpu")
        attention_mechanism = False

        expected_inferred_model_type = "bpemb"

        actual_inferred_model_type, _ = infer_model_type(checkpoint_weights, attention_mechanism)

        self.assertEqual(expected_inferred_model_type, actual_inferred_model_type)


if __name__ == "__main__":
    unittest.main()
