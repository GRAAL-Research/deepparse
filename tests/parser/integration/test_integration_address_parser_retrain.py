# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods
import os
import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call, ANY

import torch
from poutyne import Callback

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser
from tests.parser.integration.base_integration import AddressParserRetrainTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserIntegrationTest(AddressParserRetrainTestCase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserIntegrationTest, cls).setUpClass()
        cls.class_training_setup()

    @classmethod
    def class_training_setup(cls):
        file_extension = "p"
        training_dataset_name = "sample_noisy_data"
        test_dataset_name = "test_sample_data"
        download_from_url(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)
        download_from_url(test_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.training_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension))
        cls.test_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, test_dataset_name + "." + file_extension))

    # Retrain API tests
    def test_givenAFasttextAddressParser_whenRetrain_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainMultipleEpochs_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_three_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithConfig_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithConfigWithCallbacks_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        callback_mock = MagicMock(spec=Callback)
        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            callbacks=[callback_mock],
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

        callback_train_start_call = [call.on_train_begin({})]
        callback_mock.assert_has_calls(callback_train_start_call)
        callback_train_end_call = [call.on_train_end({})]
        callback_mock.assert_has_calls(callback_train_end_call)
        callback_mock.assert_not_called()

    def test_givenAFasttextLightAddressParser_whenRetrain_thenTrainingDoesNotOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        with self.assertRaises(ValueError):
            _ = address_parser.retrain(self.training_container,
                                       self.a_train_ratio,
                                       epochs=self.a_single_epoch,
                                       batch_size=self.a_batch_size,
                                       num_workers=self.a_number_of_workers,
                                       logging_path=self.a_checkpoints_saving_dir)

    def test_givenABPEmbAddressParser_whenRetrain_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainMultipleEpochs_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_three_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithConfig_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithConfigWithCallbacks_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        callback_mock = MagicMock(spec=Callback)
        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            callbacks=[callback_mock],
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

        callback_train_start_call = [call.on_train_begin({})]
        callback_mock.assert_has_calls(callback_train_start_call)
        callback_train_end_call = [call.on_train_end({})]
        callback_mock.assert_has_calls(callback_train_end_call)
        callback_mock.assert_not_called()

    # Test API tests
    def test_givenAFasttextAddressParser_whenTest_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestMultipleEpochs_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfig_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigWithCallbacks_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     callbacks=[callback_mock],
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

        callback_test_start_call = [call.on_test_begin({})]
        callback_mock.assert_has_calls(callback_test_start_call)
        callback_test_end_call = [
            call.on_test_end({
                "time": ANY,
                "test_loss": performance_after_test["test_loss"],
                "test_accuracy": performance_after_test["test_accuracy"]
            })
        ]
        callback_mock.assert_has_calls(callback_test_end_call)
        callback_mock.assert_not_called()

    def test_givenAFasttextAddressParser_whenTestWithIntCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=self.a_single_epoch)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithLastCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="last")

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithFasttextCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="fasttext")

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithStrCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=self.fasttext_local_path)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTest_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestMultipleEpochs_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfig_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigWithCallbacks_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     callbacks=[callback_mock],
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

        callback_test_start_call = [call.on_test_begin({})]
        callback_mock.assert_has_calls(callback_test_start_call)
        callback_test_end_call = [
            call.on_test_end({
                "time": ANY,
                "test_loss": performance_after_test["test_loss"],
                "test_accuracy": performance_after_test["test_accuracy"]
            })
        ]
        callback_mock.assert_has_calls(callback_test_end_call)
        callback_mock.assert_not_called()

    def test_givenABPEmbAddressParser_whenTestWithIntCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=self.a_single_epoch)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithLastCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="last")

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithBPEmbCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="bpemb")

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithStrCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=self.bpemb_local_path)

        self.assertIsNotNone(performance_after_test)


if __name__ == "__main__":
    unittest.main()
