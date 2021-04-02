# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=W0613, too-many-arguments
import os
import unittest
from unittest.mock import patch, call

import torch

from deepparse.parser import AddressParser, DatasetContainer, nll_loss, accuracy
from tests.parser.base import AddressParserPredictTestCase

batch_size = 32


class ADataContainer(DatasetContainer):

    def __init__(self, ):
        super().__init__()
        self.data = (torch.rand(batch_size, 1), torch.rand(batch_size, 1))


class AddressParserRetrainTest(AddressParserPredictTestCase):
    # pylint: disable=too-many-public-methods
    @classmethod
    def setUpClass(cls):
        super(AddressParserRetrainTest, cls).setUpClass()
        cls.a_device = "cpu"

    def setUp(self):
        super().setUp()
        self.a_train_ratio = 0.8
        self.a_batch_size = batch_size
        self.a_epoch_number = 1
        self.a_number_of_workers = 1
        self.a_learning_rate = 0.01
        self.a_callbacks_list = []
        self.a_seed = 42
        self.a_logging_path = "./ckpts"
        self.a_torch_device = torch.device(self.a_device)

        self.a_loss_function = nll_loss
        self.a_list_of_batch_metrics = [accuracy]

        self.mocked_data_container = ADataContainer()

        self.a_best_checkpoint = "best"

        self.verbose = False

        # to create the dirs for dumping the prediction tags since we mock Poutyne that usually will do it
        os.makedirs(self.a_logging_path, exist_ok=True)

    def tearDown(self) -> None:
        # cleanup after the tests
        path = os.path.join(self.a_logging_path, "./prediction_tags.p")
        if os.path.exists(path):
            os.remove(path)

        os.rmdir(self.a_logging_path)

    def address_parser_retrain_call(self):
        self.address_parser.retrain(self.mocked_data_container,
                                    self.a_train_ratio,
                                    self.a_batch_size,
                                    self.a_epoch_number,
                                    num_workers=self.a_number_of_workers,
                                    learning_rate=self.a_learning_rate,
                                    callbacks=self.a_callbacks_list,
                                    seed=self.a_seed,
                                    logging_path=self.a_logging_path)

    def assert_experiment_retrain(self, experiment_mock, model_mock, optimizer_mock):
        experiment_mock.assert_called_with(self.a_logging_path,
                                           model_mock(),
                                           device=self.a_torch_device,
                                           optimizer=optimizer_mock(),
                                           loss_function=self.a_loss_function,
                                           batch_metrics=self.a_list_of_batch_metrics)

    def assert_experiment_train_method_is_call(self, dataloader_mock, experiment_mock):
        train_call = [
            call().train(dataloader_mock(),
                         valid_generator=dataloader_mock(),
                         epochs=self.a_epoch_number,
                         seed=self.a_seed,
                         callbacks=[])
        ]
        experiment_mock.assert_has_calls(train_call)

    def address_parser_test_call(self):
        self.address_parser.test(self.mocked_data_container,
                                 self.a_batch_size,
                                 num_workers=self.a_number_of_workers,
                                 callbacks=self.a_callbacks_list,
                                 seed=self.a_seed,
                                 logging_path=self.a_logging_path,
                                 checkpoint=self.a_best_checkpoint)

    def assert_experiment_test(self, experiment_mock, model_mock):
        experiment_mock.assert_called_with(self.a_logging_path,
                                           model_mock(),
                                           device=self.a_torch_device,
                                           loss_function=self.a_loss_function,
                                           batch_metrics=self.a_list_of_batch_metrics)

    def assert_experiment_test_method_is_call(self, dataloader_mock, experiment_mock):
        test_call = [call().test(dataloader_mock(), seed=self.a_seed, callbacks=[], checkpoint=self.a_best_checkpoint)]
        experiment_mock.assert_has_calls(test_call)

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
                                                                      experiment_mock):
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

    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrain_thenRaiseError(self, embeddings_model_mock, vectorizer_model_mock,
                                                         data_padding_mock, model_mock, data_transform_mock,
                                                         optimizer_mock, experiment_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        optimizer_mock.assert_called_with(model_mock().parameters(), self.a_learning_rate)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrain_thenInstantiateExperimentProperly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenRetrain_thenInstantiateDataLoaderAndTrainProperly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        self.assert_experiment_train_method_is_call(dataloader_mock, experiment_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrain_thenInstantiateExperimentProperly(self, embeddings_model_mock,
                                                                            vectorizer_model_mock, data_padding_mock,
                                                                            model_mock, data_transform_mock,
                                                                            optimizer_mock, experiment_mock,
                                                                            dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        self.assert_experiment_retrain(experiment_mock, model_mock, optimizer_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenRetrain_thenInstantiateDataLoaderAndTrainProperly(self, embeddings_model_mock,
                                                                                    vectorizer_model_mock,
                                                                                    data_padding_mock, model_mock,
                                                                                    data_transform_mock, optimizer_mock,
                                                                                    experiment_mock, dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_retrain_call()

        self.assert_experiment_train_method_is_call(dataloader_mock, experiment_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenTest_thenInstantiateExperimentProperly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, model_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenTest_thenInstantiateDataLoaderAndTestProperly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(dataloader_mock, experiment_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenTest_thenInstantiateExperimentProperly(self, embeddings_model_mock,
                                                                         vectorizer_model_mock, data_padding_mock,
                                                                         model_mock, data_transform_mock,
                                                                         optimizer_mock, experiment_mock,
                                                                         dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, model_mock)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenTest_thenInstantiateDataLoaderAndTestProperly(self, embeddings_model_mock,
                                                                                vectorizer_model_mock,
                                                                                data_padding_mock, model_mock,
                                                                                data_transform_mock, optimizer_mock,
                                                                                experiment_mock, dataloader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(dataloader_mock, experiment_mock)


if __name__ == "__main__":
    unittest.main()
