# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-arguments
import unittest
from unittest import skipIf
from unittest.mock import patch, call

import torch

from deepparse.metrics import accuracy, nll_loss
from deepparse.parser import AddressParser
from tests.parser.base import AddressParserPredictTestCase
from tests.tools import BATCH_SIZE, ADataContainer


class AddressParserRetrainTest(AddressParserPredictTestCase):
    # pylint: disable=too-many-public-methods
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

        cls.a_fasttext_path = "fasttext"
        cls.a_bpemb_path = "bpemb"

        cls.verbose = False

    def address_parser_test_call(self):
        self.address_parser.test(self.mocked_data_container,
                                 self.a_batch_size,
                                 num_workers=self.a_number_of_workers,
                                 callbacks=self.a_callbacks_list,
                                 seed=self.a_seed)

    def assert_experiment_test(self, experiment_mock, model_mock, device):
        experiment_mock.assert_called_with(
            "./checkpoint",  # We always use this as default logging dir.
            model_mock(),
            device=device,
            # For a reason I don't understand if I use self.nll_loss and set it in the
            # class setup, it return a bound method for the nll_loss but it work for
            # the accuracy. So fuck it, here a fix.
            loss_function=nll_loss,
            batch_metrics=[accuracy],
            logging=False)

    def assert_experiment_test_method_is_call(self, data_loader_mock, experiment_mock, verbose):
        test_call = [call().test(data_loader_mock(), seed=self.a_seed, callbacks=[], verbose=verbose)]
        experiment_mock.assert_has_calls(test_call)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenTestCPU_thenInstantiateExperimentProperly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, model_mock, device=self.a_device)

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
    def test_givenAFasttextModel_whenTestGPU_thenInstantiateExperimentProperly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_torch_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, model_mock, device=self.a_torch_device)

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
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock):
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModel_whenTestVerbose_thenInstantiateWithVerbose(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock, model_mock,
            data_transform_mock, optimizer_mock, experiment_mock, data_loader_mock):
        verbose = True
        self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                            device=self.a_device,
                                            verbose=verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=verbose)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenTestCPU_thenInstantiateExperimentProperly(self, embeddings_model_mock,
                                                                            vectorizer_model_mock, data_padding_mock,
                                                                            model_mock, data_transform_mock,
                                                                            optimizer_mock, experiment_mock,
                                                                            data_loader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, model_mock, device=self.a_device)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenABPEmbModel_whenTestGPU_thenInstantiateExperimentProperly(self, embeddings_model_mock,
                                                                            vectorizer_model_mock, data_padding_mock,
                                                                            model_mock, data_transform_mock,
                                                                            optimizer_mock, experiment_mock,
                                                                            data_loader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_torch_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, model_mock, device=self.a_torch_device)

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
                                                                                experiment_mock, data_loader_mock):
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                            device=self.a_device,
                                            verbose=self.verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.DataTransform")
    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModel_whenTestVerboseTrue_thenInstantiateWithVerbose(self, embeddings_model_mock,
                                                                             vectorizer_model_mock, data_padding_mock,
                                                                             model_mock, data_transform_mock,
                                                                             optimizer_mock, experiment_mock,
                                                                             data_loader_mock):
        verbose = True
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type, device=self.a_device, verbose=verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=verbose)


if __name__ == "__main__":
    unittest.main()
