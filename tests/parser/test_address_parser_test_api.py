# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, too-many-arguments
import os
import platform
import unittest
from unittest import skipIf
from unittest.mock import MagicMock, patch, call

import torch

from deepparse.errors import FastTextModelError
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
        cls.a_number_of_workers = 0
        cls.a_learning_rate = 0.01
        cls.a_callbacks_list = []
        cls.a_seed = 42
        cls.a_torch_device = torch.device("cuda:0")

        cls.mocked_data_container = ADataContainer()

        cls.a_fasttext_path = "fasttext"
        cls.a_bpemb_path = "bpemb"

        cls.verbose = False

    def setUp(self):
        self.model_mock = MagicMock()

    def address_parser_test_call(self, dataset_container=None, num_workers=None):
        if dataset_container is None:
            dataset_container = self.mocked_data_container
            # To handle by default for most of the tests.

        if num_workers is None:
            num_workers = self.a_number_of_workers

        self.address_parser.test(
            dataset_container,
            self.a_batch_size,
            num_workers=num_workers,
            callbacks=self.a_callbacks_list,
            seed=self.a_seed,
        )

    def assert_experiment_test(self, experiment_mock, model_mock, device):
        experiment_mock.assert_called_with(
            "./checkpoint",  # We always use this as default logging dir.
            model_mock,
            device=device,
            # For a reason I don't understand if I use self.nll_loss and set it in the
            # class setup, it return a bound method for the nll_loss but it work for
            # the accuracy. So fuck it, here a fix.
            loss_function=nll_loss,
            batch_metrics=[accuracy],
            logging=False,
        )

    def assert_experiment_test_method_is_call(self, data_loader_mock, experiment_mock, verbose):
        test_call = [call().test(data_loader_mock(), seed=self.a_seed, callbacks=[], verbose=verbose)]
        experiment_mock.assert_has_calls(test_call)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenAFasttextModel_whenTestCPU_thenInstantiateExperimentProperly(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        model_factory_mock().create.return_value = self.model_mock
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, self.model_mock, device=self.a_device)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    @skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
    def test_givenAFasttextModel_whenTestGPU_thenInstantiateExperimentProperly(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        model_factory_mock().create.return_value = self.model_mock
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_torch_device,
            verbose=self.verbose,
        )
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, self.model_mock, device=self.a_torch_device)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenAFasttextModel_whenTest_thenInstantiateDataLoaderAndTestProperly(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenAFasttextModel_whenTestVerbose_thenInstantiateWithVerbose(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        verbose = True
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type, device=self.a_device, verbose=verbose
        )
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=verbose)

    @skipIf(platform.system() != "Windows", "Integration test on Windows env.")
    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenAFasttextModelOnWindows_whenTestVerboseWithNumWorkerGT0_thenRaiseError(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        verbose = True
        self.address_parser = AddressParser(
            model_type=self.a_fasttext_model_type, device=self.a_device, verbose=verbose
        )
        a_num_worker_gt_0 = 1
        with self.assertRaises(FastTextModelError):
            self.address_parser_test_call(num_workers=a_num_worker_gt_0)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenABPEmbModel_whenTestCPU_thenInstantiateExperimentProperly(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        model_factory_mock().create.return_value = self.model_mock
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, self.model_mock, device=self.a_device)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    @skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
    def test_givenABPEmbModel_whenTestGPU_thenInstantiateExperimentProperly(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        model_factory_mock().create.return_value = self.model_mock
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_torch_device,
            verbose=self.verbose,
        )
        self.address_parser_test_call()

        self.assert_experiment_test(experiment_mock, self.model_mock, device=self.a_torch_device)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenABPEmbModel_whenTest_thenInstantiateDataLoaderAndTestProperly(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.DataLoader")
    @patch("deepparse.parser.address_parser.Experiment")
    @patch("deepparse.parser.address_parser.SGD")
    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenABPEmbModel_whenTestVerboseTrue_thenInstantiateWithVerbose(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
        optimizer_mock,
        experiment_mock,
        data_loader_mock,
    ):
        verbose = True
        self.address_parser = AddressParser(model_type=self.a_bpemb_model_type, device=self.a_device, verbose=verbose)
        self.address_parser_test_call()

        self.assert_experiment_test_method_is_call(data_loader_mock, experiment_mock, verbose=verbose)

    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenNotTrainingDataContainer_thenRaiseValueError(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        mocked_data_container = ADataContainer(is_training_container=False)
        with self.assertRaises(ValueError):
            self.address_parser_test_call(dataset_container=mocked_data_container)

    @patch("deepparse.parser.address_parser.ModelFactory")
    @patch("deepparse.parser.address_parser.EmbeddingsModelFactory")
    @patch("deepparse.parser.address_parser.VectorizerFactory")
    @patch("deepparse.parser.address_parser.DataProcessorFactory")
    @patch("deepparse.parser.address_parser.DataPadder")
    def test_givenNotADataContainer_thenRaiseValueError(
        self,
        data_padder_mock,
        data_processor_factory_mock,
        vectorizer_factory_mock,
        embeddings_model_factory_mock,
        model_factory_mock,
    ):
        self.address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_device,
            verbose=self.verbose,
        )
        not_a_dataset_container = []
        with self.assertRaises(ValueError):
            self.address_parser_test_call(dataset_container=not_a_dataset_container)

        not_a_dataset_container = {}
        with self.assertRaises(ValueError):
            self.address_parser_test_call(dataset_container=not_a_dataset_container)


if __name__ == "__main__":
    unittest.main()
