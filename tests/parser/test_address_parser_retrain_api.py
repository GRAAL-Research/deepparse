# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-arguments, too-many-public-methods
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import skipIf
from unittest.mock import patch, call, MagicMock

import torch

from deepparse.converter import TagsConverter
from deepparse.metrics import nll_loss, accuracy
from deepparse.parser import AddressParser
from tests.parser.base import AddressParserPredictTestCase
from tests.tools import BATCH_SIZE, ADataContainer


class AddressParserRetrainTest(AddressParserPredictTestCase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserRetrainTest, cls).setUpClass()
        cls.a_device = torch.device("cpu")

        cls.a_train_ratio = 0.8
        cls.a_batch_size = BATCH_SIZE
        cls.a_epoch_number = 1
        cls.a_number_of_workers = 1
        cls.a_learning_rate = 0.01
        cls.a_callbacks_list = []
        cls.a_seed = 42
        cls.a_torch_device = torch.device("cuda:0")

        cls.mocked_data_container = ADataContainer()

        cls.a_best_checkpoint = "best"

        cls.verbose = False

        cls.address_components = {"ATag": 0, "AnotherTag": 1, "EOS": 2}

    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.a_logging_path = os.path.join(self.temp_dir_obj.name, "ckpts")
        self.saving_template_path = os.path.join(self.a_logging_path, "retrained_{}_address_parser.ckpt")

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def address_parser_retrain_call(self, prediction_tags=None):
        self.address_parser.retrain(self.mocked_data_container,
                                    self.a_train_ratio,
                                    self.a_batch_size,
                                    self.a_epoch_number,
                                    num_workers=self.a_number_of_workers,
                                    learning_rate=self.a_learning_rate,
                                    callbacks=self.a_callbacks_list,
                                    seed=self.a_seed,
                                    logging_path=self.a_logging_path,
                                    prediction_tags=prediction_tags)

    def assert_experiment_retrain(self, experiment_mock, model_mock, optimizer_mock, device):
        experiment_mock.assert_called_with(
            self.a_logging_path,
            model_mock(),
            device=device,
            optimizer=optimizer_mock(),
            # For a reason I don't understand if I use self.nll_loss and set it in the
            # class setup, it return a bound method for the nll_loss but it work for
            # the accuracy. So fuck it, here a fix.
            loss_function=nll_loss,
            batch_metrics=[accuracy])

    def assert_experiment_train_method_is_call(self, data_loader_mock, experiment_mock):
        train_call = [
            call().train(data_loader_mock(),
                         valid_generator=data_loader_mock(),
                         epochs=self.a_epoch_number,
                         seed=self.a_seed,
                         callbacks=[],
                         verbose=self.verbose,
                         disable_tensorboard=True)
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
    def test_givenAFasttextModel_whenRetrain_thenInstantiateOptimizer(self, download_weights_mock,
                                                                      embeddings_model_mock, vectorizer_model_mock,
                                                                      data_padding_mock, model_mock,
                                                                      data_transform_mock, optimizer_mock,
                                                                      experiment_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        optimizer_mock.assert_called_with(model_mock().parameters(), self.a_learning_rate)

    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    def test_givenAFasttextMagnitudeModel_whenRetrain_thenRaiseError(self, download_weights_mock, embeddings_model_mock,
                                                                     vectorizer_model_mock, data_padding_mock,
                                                                     mock_model, experiment_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)

        with self.assertRaises(ValueError):
            self.address_parser_retrain_call()

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
    def test_givenAFasttextModel_whenRetrain_thenSaveModelProperly(self, download_weights_mock, embeddings_model_mock,
                                                                   vectorizer_model_mock, data_padding_mock, model_mock,
                                                                   data_transform_mock, optimizer_mock, experiment_mock,
                                                                   data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    'address_tagger_model': experiment_mock().model.network.state_dict(),
                    'model_type': self.a_fasttext_model_type
                }, saving_model_path)
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
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
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
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_torch_device,
                                            verbose=self.verbose)
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
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call(prediction_tags=self.address_components)

        saving_model_path = self.saving_template_path.format(self.a_fasttext_model_type)
        save_call = [
            call(
                {
                    'address_tagger_model': experiment_mock().model.network.state_dict(),
                    'prediction_tags': self.address_components,
                    'model_type': self.a_fasttext_model_type
                }, saving_model_path)
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
    def test_givenABPEmbModel_whenRetrainCPU_thenInstantiateExperimentProperly(self, embeddings_model_mock,
                                                                               vectorizer_model_mock, data_padding_mock,
                                                                               model_mock, data_transform_mock,
                                                                               optimizer_mock, experiment_mock,
                                                                               data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
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
    def test_givenABPEmbModel_whenRetrainGPU_thenInstantiateExperimentProperly(self, embeddings_model_mock,
                                                                               vectorizer_model_mock, data_padding_mock,
                                                                               model_mock, data_transform_mock,
                                                                               optimizer_mock, experiment_mock,
                                                                               data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_torch_device,
                                            verbose=self.verbose)
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
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock, data_transform_mock,
            optimizer_mock, experiment_mock, data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
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
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock, data_transform_mock,
            optimizer_mock, experiment_mock, data_loader_mock, torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_torch_device,
                                            verbose=self.verbose)
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
    def test_givenABPEmbModel_whenRetrainWithUserTags_thenSaveTagsDict(self, embeddings_model_mock,
                                                                       vectorizer_model_mock, data_padding_mock,
                                                                       model_mock, data_transform_mock, optimizer_mock,
                                                                       experiment_mock, data_loader_mock,
                                                                       torch_save_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call(prediction_tags=self.address_components)
        saving_model_path = self.saving_template_path.format(self.a_bpemb_model_type)
        save_call = [
            call(
                {
                    'address_tagger_model': experiment_mock().model.network.state_dict(),
                    'prediction_tags': self.address_components,
                    'model_type': self.a_bpemb_model_type
                }, saving_model_path)
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
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_patch, data_transform_mock,
            optimizer_mock, experiment_mock, data_loader_mock, torch_save_mock, tags_converter_patch):
        # we test with BPEmb but fasttext would give same results
        model_mock = MagicMock()
        model_mock.same_output_dim.return_value = False
        model_patch.return_value = model_mock

        tags_converter_mock = MagicMock(spec=TagsConverter)
        tags_converter_patch.return_value = tags_converter_mock

        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call(prediction_tags=self.address_components)
        new_dim_call = [call.handle_new_output_dim(tags_converter_mock.dim)]

        model_mock.assert_has_calls(new_dim_call)


if __name__ == "__main__":
    unittest.main()
