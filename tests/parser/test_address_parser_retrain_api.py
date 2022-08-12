# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-arguments, too-many-public-methods, protected-access, too-many-lines

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import platform
import unittest
from tempfile import TemporaryDirectory
from unittest import skipIf
from unittest.mock import patch, call, MagicMock

import torch

from deepparse.converter import TagsConverter
from deepparse.metrics import nll_loss, accuracy
from deepparse.parser import AddressParser
from tests.parser.base import AddressParserPredictTestCase
from tests.tools import BATCH_SIZE, ADataContainer, create_file


class AddressParserRetrainTest(AddressParserPredictTestCase):
    @classmethod
    def setUpClass(cls):
        super(AddressParserRetrainTest, cls).setUpClass()
        cls.a_device = torch.device("cpu")

        cls.a_train_ratio = 0.8
        cls.a_batch_size = BATCH_SIZE
        cls.a_epoch_number = 1
        cls.a_learning_rate = 0.01
        cls.a_callbacks_list = []
        cls.a_seed = 42
        cls.a_torch_device = torch.device("cuda:0")

        cls.mocked_data_container = ADataContainer()

        cls.a_best_checkpoint = "best"

        cls.verbose = False

        cls.address_components = {"ATag": 0, "AnotherTag": 1, "EOS": 2}

        cls.seq2seq_params = {"encoder_hidden_size": 512, "decoder_hidden_size": 512}

        cls.a_named_parser_name = "AModelName"

    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.a_logging_path = os.path.join(self.temp_dir_obj.name, "ckpts")
        os.makedirs(self.a_logging_path)
        self.saving_template_path = os.path.join(self.a_logging_path, "retrained_{}_address_parser.ckpt")

    def populate_directory(self):
        create_file(
            os.path.join(self.a_logging_path, "retrained_fasttext_address_parser.ckpt"),
            "a content",
        )

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def address_parser_retrain_call(
        self,
        train_dataset_container=None,
        val_dataset_container=None,
        train_ratio=None,  # None to handle default test case
        prediction_tags=None,
        seq2seq_params=None,
        layers_to_freeze=None,
        name_of_the_retrain_parser=None,
        num_workers=None,
    ):
        if num_workers is None:
            # AddressParser default num_workers settings is 1
            # But, we change it to 0 for Windows OS to allow test to pass since it fail (volontairy)
            # at greater than 0 due to parallelism pickle error
            if platform.system().lower() == "windows":
                num_workers = 0  # Default setting is 1, but We set it to zero to allow Windows tests to pass
            else:
                num_workers = 1

        if train_dataset_container is None:
            train_dataset_container = self.mocked_data_container
            # To handle by default for most of the tests.

        if train_ratio is None:
            train_ratio = self.a_train_ratio

        self.address_parser.retrain(
            train_dataset_container,
            val_dataset_container,
            train_ratio,
            self.a_batch_size,
            self.a_epoch_number,
            num_workers=num_workers,
            learning_rate=self.a_learning_rate,
            callbacks=self.a_callbacks_list,
            seed=self.a_seed,
            logging_path=self.a_logging_path,
            prediction_tags=prediction_tags,
            seq2seq_params=seq2seq_params,
            layers_to_freeze=layers_to_freeze,
            name_of_the_retrain_parser=name_of_the_retrain_parser,
        )

    def assert_experiment_retrain(self, experiment_mock, model_mock, optimizer_mock, device):
        experiment_mock.assert_called_with(
            self.a_logging_path,
            model_mock(),
            device=device,
            optimizer=optimizer_mock(),
            # For a reason I don"t understand if I use self.nll_loss and set it in the
            # class setup, it return a bound method for the nll_loss but it work for
            # the accuracy. So fuck it, here a fix.
            loss_function=nll_loss,
            batch_metrics=[accuracy],
        )

    def assert_experiment_train_method_is_call(self, data_loader_mock, experiment_mock):
        train_call = [
            call().train(
                data_loader_mock(),
                valid_generator=data_loader_mock(),
                epochs=self.a_epoch_number,
                seed=self.a_seed,
                callbacks=[],
                verbose=self.verbose,
                disable_tensorboard=True,
            )
        ]
        experiment_mock.assert_has_calls(train_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrain_thenInstantiateOptimizer(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        optimizer_mock.assert_called_with(model_mock().parameters(), self.a_learning_rate)

    @patch("deepparse.tools.poutyne")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAModel_whenRetrainWithPoutyneBefore18_thenPrintMessage(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        torch_save_mock,
        poutyne_mock,
    ):
        poutyne_mock.version.__version__ = "1.7"
        self._capture_output()
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        actual = self.test_out.getvalue()
        expected = (
            "You are using a older version of Poutyne that does not support properly error management."
            " Due to that, we cannot show retrain progress. To fix that, update Poutyne to the newest version.\n"
        )

        self.assertEqual(actual, expected)

    @patch("deepparse.tools.poutyne")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAModel_whenRetrainWithPoutyneAfter17_thenDoNotPrintMessage(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        torch_save_mock,
        poutyne_mock,
    ):
        poutyne_mock.version.__version__ = "1.8"
        self._capture_output()
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        actual = self.test_out.getvalue()

        expected = ""
        self.assertEqual(actual, expected)

        not_expected = (
            "You are using a older version of Poutyne that does not support properly error management."
            " Due to that, we cannot show retrain progress. To fix that, update Poutyne to the newest "
            "version.\n"
        )

        self.assertNotRegex(actual, not_expected)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch(
        "deepparse.parser.address_parser.Experiment",
        **{"return_value.train.side_effect": RuntimeError()},
    )
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModelDirectoryWithOtherRetrainModel_whenRetrain_thenRaiseError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        torch_save_mock,
    ):
        self.populate_directory()
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        with self.assertRaises(ValueError):
            self.address_parser_retrain_call()

        expect_error_message = (
            f"You are currently training a different fasttext version from the one in"
            f" the {self.a_logging_path}. Verify version."
        )
        try:
            self.address_parser_retrain_call()
        except ValueError as actual_error_message:
            self.assertEqual(actual_error_message.args[0], expect_error_message)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch(
        "deepparse.parser.address_parser.Experiment",
        **{"return_value.train.side_effect": RuntimeError()},
    )
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenABPEmbModelDirectoryWithOtherFastTextRetrainModel_whenRetrain_thenRaiseError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        torch_save_mock,
    ):
        self.populate_directory()
        self.address_parser = AddressParser(
            model_type=self.a_best_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        with self.assertRaises(ValueError):
            self.address_parser_retrain_call()

        expect_error_message = (
            f"You are currently training a bpemb in the directory {self.a_logging_path} where a "
            f"different retrained fasttext is currently his. Thus, the loading of the model is "
            f"failing. Change directory to retrain the bpemb."
        )
        try:
            self.address_parser_retrain_call()
        except ValueError as actual_error_message:
            self.assertEqual(actual_error_message.args[0], expect_error_message)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch(
        "deepparse.parser.address_parser.Experiment",
        **{"return_value.train.side_effect": RuntimeError("Error during training")},
    )
    @patch("deepparse.parser.address_parser.os.listdir", return_value=[])
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAModelDirectoryWithoutOtherRetrainModel_whenRetrainRaisesRuntimeError_thenReRaiseError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        torch_save_mock,
        os_mock,
    ):
        # os_mock.listdir.return_value = []
        self.address_parser = AddressParser(
            model_type=self.a_best_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        with self.assertRaises(RuntimeError):
            self.address_parser_retrain_call()

        expect_error_message = "Error during training"
        try:
            self.address_parser_retrain_call()
        except RuntimeError as actual_error_message:
            self.assertEqual(actual_error_message.args[0], expect_error_message)

    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    def test_givenAFasttextMagnitudeModel_whenRetrain_thenRaiseError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        mock_model,
        experiment_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_light_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        with self.assertRaises(ValueError):
            self.address_parser_retrain_call()

    @patch("deepparse.parser.address_parser.platform")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFastTextLikeModelOnWindowsOS_whenRetrainWithNumWorkersGT0_thenReRaiseError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        platform_mock,
    ):
        # OS equal Windows
        platform_mock.system().__eq__.return_value = True

        self.address_parser = AddressParser(
            model_type=self.a_fastest_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        num_workers_gt_0 = 1
        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(num_workers=num_workers_gt_0)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrain_thenSaveModelProperly(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()
        expected_named_parser_name = "FastText"
        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_fasttext_model_type,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenFastTextModel_whenRetrainCPU_thenInstantiateExperimentProperly(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock, device=self.a_device)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenFastTextModel_whenRetrainGPU_thenInstantiateExperimentProperly(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_torch_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock, device=self.a_torch_device)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrainWithUserTags_thenSaveTagsDict(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(prediction_tags=self.address_components)

        expected_named_parser_name = "FastTextModifiedPredictionTags"

        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "prediction_tags": self.address_components,
                    "model_type": self.a_fasttext_model_type,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrainWithNewParams_thenModelFactoryIsCalled(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        model_factory_mock = MagicMock()
        self.address_parser._model_factory = model_factory_mock
        self.address_parser_retrain_call(seq2seq_params=self.seq2seq_params)

        model_factory_mock.assert_called()

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrainWithNewParams_thenSaveNewParamsDict(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(seq2seq_params=self.seq2seq_params)

        expected_named_parser_name = "FastTextModifiedSeq2SeqConfiguration"

        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_fasttext_model_type,
                    "seq2seq_params": self.seq2seq_params,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrainWithNewParamsAndNewTags_thenSaveNewParamsDictAndParams(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(prediction_tags=self.address_components, seq2seq_params=self.seq2seq_params)

        expected_named_parser_name = "FastTextModifiedPredictionTagsModifiedSeq2SeqConfiguration"

        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_fasttext_model_type,
                    "seq2seq_params": self.seq2seq_params,
                    "prediction_tags": self.address_components,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrainWithNewParamsAndNewTagsAndFreezeLayers_thenSaveNewParamsDictAndParams(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(
            prediction_tags=self.address_components, seq2seq_params=self.seq2seq_params, layers_to_freeze="encoder"
        )

        expected_named_parser_name = "FastTextModifiedPredictionTagsModifiedSeq2SeqConfigurationFreezedLayerEncoder"

        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_fasttext_model_type,
                    "seq2seq_params": self.seq2seq_params,
                    "prediction_tags": self.address_components,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainCPU_thenInstantiateExperimentProperly(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock, device=self.a_device)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenABPEmbModel_whenRetrainGPU_thenInstantiateExperimentProperly(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_torch_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock, device=self.a_torch_device)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainCPU_thenInstantiateDataLoaderAndTrainProperly(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock, device=self.a_device)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenABPEmbModel_whenRetrainGPU_thenInstantiateDataLoaderAndTrainProperly(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_torch_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock, device=self.a_torch_device)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrain_thenSaveModelProperly(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call()
        expected_named_parser_name = "BPEmb"
        saving_model_path = self.saving_template_path.format(self.a_bpemb_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_bpemb_model_type,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainWithUserTags_thenSaveTagsDict(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        self.address_parser_retrain_call(prediction_tags=self.address_components)

        expected_named_parser_name = "BPEmbModifiedPredictionTags"

        saving_model_path = self.saving_template_path.format(self.a_bpemb_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "prediction_tags": self.address_components,
                    "model_type": self.a_bpemb_model_type,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainWithNewParams_thenModelFactoryIsCalled(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        model_factory_mock = MagicMock()
        self.address_parser._model_factory = model_factory_mock
        self.address_parser_retrain_call(seq2seq_params=self.seq2seq_params)

        model_factory_mock.assert_called()

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainWithNewParams_thenSaveNewParamsDict(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(seq2seq_params=self.seq2seq_params)

        expected_named_parser_name = "BPEmbModifiedSeq2SeqConfiguration"

        saving_model_path = self.saving_template_path.format(self.a_bpemb_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_bpemb_model_type,
                    "seq2seq_params": self.seq2seq_params,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainWithNewParamsAndNewTags_thenSaveNewParamsDictAndParams(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(prediction_tags=self.address_components, seq2seq_params=self.seq2seq_params)

        expected_named_parser_name = "BPEmbModifiedPredictionTagsModifiedSeq2SeqConfiguration"

        saving_model_path = self.saving_template_path.format(self.a_bpemb_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_bpemb_model_type,
                    "seq2seq_params": self.seq2seq_params,
                    "prediction_tags": self.address_components,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrainWithNewParamsAndNewTagsAndFreezeLayers_thenSaveNewParamsDictAndParams(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(
            prediction_tags=self.address_components, seq2seq_params=self.seq2seq_params, layers_to_freeze="encoder"
        )

        expected_named_parser_name = "BPEmbModifiedPredictionTagsModifiedSeq2SeqConfigurationFreezedLayerEncoder"

        saving_model_path = self.saving_template_path.format(self.a_bpemb_model_type)
        save_call = [
            call(
                {
                    "address_tagger_model": experiment_mock().model.network.state_dict(),
                    "model_type": self.a_bpemb_model_type,
                    "seq2seq_params": self.seq2seq_params,
                    "prediction_tags": self.address_components,
                    "named_parser": expected_named_parser_name,
                },
                saving_model_path,
            )
        ]

        torch_save_mock.assert_has_calls(save_call)

    @patch("deepparse.parser.address_parser.TagsConverter")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenNewPredictionTagsNewDimSize_thenHandleNewOutputDimProperly(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_patch,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
        tags_converter_patch,
    ):
        # we test with BPEmb but fasttext would give same results
        model_mock = MagicMock()
        model_mock.same_output_dim.return_value = False
        model_patch.return_value = model_mock

        tags_converter_mock = MagicMock(spec=TagsConverter)
        tags_converter_patch.return_value = tags_converter_mock

        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_retrain_call(prediction_tags=self.address_components)
        new_dim_call = [call.handle_new_output_dim(tags_converter_mock.dim)]

        model_mock.assert_has_calls(new_dim_call)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenNotTrainingDataContainer_thenRaiseValueError(
        self,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_patch,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        mocked_data_container = ADataContainer(is_training_container=False)

        a_number_of_workers = 0  # We set it to 0 to allow Windows test to also pass

        with self.assertRaises(ValueError):
            self.address_parser.retrain(
                mocked_data_container,
                train_ratio=self.a_train_ratio,
                batch_size=self.a_batch_size,
                epochs=self.a_epoch_number,
                num_workers=a_number_of_workers,
                learning_rate=self.a_learning_rate,
                callbacks=self.a_callbacks_list,
                seed=self.a_seed,
                logging_path=self.a_logging_path,
            )

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenNotFreezeLayers_thenFreezeLayerMethodNotCalled(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        freeze_model_params_method_mock = MagicMock()
        self.address_parser._freeze_model_params = freeze_model_params_method_mock
        self.address_parser_retrain_call(layers_to_freeze=None)

        freeze_model_params_method_mock.assert_not_called()

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenFreezeLayersEncoder_thenFreezeLayerMethodCalledWithEncoder(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        freeze_model_params_method_mock = MagicMock()
        self.address_parser._freeze_model_params = freeze_model_params_method_mock
        self.address_parser_retrain_call(layers_to_freeze="encoder")

        freeze_model_params_method_mock.assert_called()
        freeze_model_params_method_mock.assert_called_with("encoder")

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenFreezeLayersDecoder_thenFreezeLayerMethodCalledWithDecoder(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        freeze_model_params_method_mock = MagicMock()
        self.address_parser._freeze_model_params = freeze_model_params_method_mock
        self.address_parser_retrain_call(layers_to_freeze="decoder")

        freeze_model_params_method_mock.assert_called()
        freeze_model_params_method_mock.assert_called_with("decoder")

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenFreezeLayersPredictionLayer_thenFreezeLayerMethodCalledWithPredictionLayer(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        freeze_model_params_method_mock = MagicMock()
        self.address_parser._freeze_model_params = freeze_model_params_method_mock
        self.address_parser_retrain_call(layers_to_freeze="prediction_layer")

        freeze_model_params_method_mock.assert_called()
        freeze_model_params_method_mock.assert_called_with("prediction_layer")

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenFreezeLayersSeq2Seq_thenFreezeLayerMethodCalledWithSeq2Seq(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        freeze_model_params_method_mock = MagicMock()
        self.address_parser._freeze_model_params = freeze_model_params_method_mock
        self.address_parser_retrain_call(layers_to_freeze="seq2seq")

        freeze_model_params_method_mock.assert_called()
        freeze_model_params_method_mock.assert_called_with("seq2seq")

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenWrongFreezeLayersName_thenRaiseValueError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(layers_to_freeze="error_in_name")

    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenRetrainSettings_whenFormattedNameParserName_thenReturnProperNaming(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
    ):
        # pylint: disable=too-many-locals, too-many-branches
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        # We set possible params type with a value
        prediction_tags_settings = [{"A dict": 1.0}, None]  # Can be a dict or a None
        seq2seq_params_settings = [{"A dict": 1.0}, None]  # Can be a dict or a None
        layers_to_freeze_settings = [None, "encoder", "decoder", "prediction_layer", "seq2seq"]  # From the doc

        # We loop all possible settings
        # Namely, not only elements settings but combinaison of settings altogether
        for prediction_tags_setting in prediction_tags_settings:
            for seq2seq_params_setting in seq2seq_params_settings:
                for layers_to_freeze_setting in layers_to_freeze_settings:
                    actual = self.address_parser._formatted_named_parser_name(
                        prediction_tags=prediction_tags_setting,
                        seq2seq_params=seq2seq_params_setting,
                        layers_to_freeze=layers_to_freeze_setting,
                    )
                    # We test if extected text is include depending on the settings
                    if prediction_tags_setting is None:
                        self.assertNotIn("ModifiedPredictionTags", actual)
                    else:
                        self.assertIn("ModifiedPredictionTags", actual)

                    if seq2seq_params_setting is None:
                        self.assertNotIn("ModifiedSeq2SeqConfiguration", actual)
                    else:
                        self.assertIn("ModifiedSeq2SeqConfiguration", actual)

                    if layers_to_freeze_setting is None:
                        self.assertNotIn("FreezedLayer", actual)
                    else:
                        self.assertIn("FreezedLayer", actual)

                        if seq2seq_params_setting == "encoder":
                            self.assertIn("encoder", actual)
                        elif seq2seq_params_setting == "decoder":
                            self.assertIn("decoder", actual)
                        elif seq2seq_params_setting == "prediction_layer":
                            self.assertIn("prediction_layer", actual)
                        elif seq2seq_params_setting == "seq2seq":
                            self.assertIn("seq2seq", actual)

    @patch("deepparse.parser.address_parser.os.path.join")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenNoneNewNamedModelName_thenSavingPathIsDefaultPathWithExtension(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
        os_path_join_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        self.address_parser_retrain_call(name_of_the_retrain_parser=None)

        os_path_join_mock.assert_called()

        default_filename = "retrained_fasttext_address_parser.ckpt"
        os_path_join_mock.assert_called_with(self.a_logging_path, default_filename)

    @patch("deepparse.parser.address_parser.os.path.join")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenNewNamedModelName_thenSavingPathIsModified(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
        os_path_join_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        self.address_parser_retrain_call(name_of_the_retrain_parser=self.a_named_parser_name)

        os_path_join_mock.assert_called()

        file_extension = ".ckpt"
        expected_filename = self.a_named_parser_name + file_extension
        os_path_join_mock.assert_called_with(self.a_logging_path, expected_filename)

    @patch("deepparse.parser.address_parser.os.path.join")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenWrongNewNamedModelName_thenRaiseValueError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
        os_path_join_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(name_of_the_retrain_parser="a_wrong_named_parser_name.ckpt")

    @patch("deepparse.parser.address_parser.os.path.join")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenNotADatasetContainer_whenRetrainCall_thenRaiseValueError(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
        os_path_join_mock,
    ):

        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        not_a_dataset_container_obj = []
        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(train_dataset_container=not_a_dataset_container_obj)

        not_a_dataset_container_obj = {}
        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(train_dataset_container=not_a_dataset_container_obj)

        # For val dataset
        not_a_dataset_container_obj = []
        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(val_dataset_container=not_a_dataset_container_obj)

        not_a_dataset_container_obj = {}
        with self.assertRaises(ValueError):
            self.address_parser_retrain_call(val_dataset_container=not_a_dataset_container_obj)

    @patch("deepparse.parser.address_parser.os.path.join")
    @patch("deepparse.parser.address_parser.torch.save")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenNotADatasetContainer_whenRetrainCallWithValDataset_thenDontUseTrainRatio(
        self,
        download_weights_mock,
        embeddings_model_mock,
        vectorizer_model_mock,
        data_padding_mock,
        model_mock,
        data_transform_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
        torch_save_mock,
        os_path_join_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )

        train_ratio_mock = MagicMock()
        self.address_parser_retrain_call(val_dataset_container=self.mocked_data_container, train_ratio=train_ratio_mock)

        train_ratio_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
