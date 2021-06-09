# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, too-many-public-methods

import os
import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call, ANY

from poutyne import Callback

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner")
class AddressParserIntegrationTestNewTags(AddressParserRetrainTestCase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserIntegrationTestNewTags, cls).setUpClass()

        file_extension = "p"
        training_dataset_name = "test_sample_data_new_prediction_tags"
        download_from_url(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.new_prediction_data_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension))

    # Retrain API tests
    def test_givenAFasttextAddressParser_whenRetrainNewTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainMultipleEpochsNewTags_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_three_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithConfigNewTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithConfigWithCallbacksNewTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        callback_mock = MagicMock(spec=Callback)
        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            callbacks=[callback_mock],
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

        callback_train_start_call = [call.on_train_begin({})]
        callback_mock.assert_has_calls(callback_train_start_call)
        callback_train_end_call = [call.on_train_end({})]
        callback_mock.assert_has_calls(callback_train_end_call)
        callback_mock.assert_not_called()

    def test_givenAFasttextLightAddressParser_whenRetrainNewTags_thenTrainingDoesNotOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        with self.assertRaises(ValueError):
            _ = address_parser.retrain(self.new_prediction_data_container,
                                       self.a_train_ratio,
                                       epochs=self.a_single_epoch,
                                       batch_size=self.a_batch_size,
                                       num_workers=self.a_number_of_workers,
                                       logging_path=self.a_checkpoints_saving_dir,
                                       prediction_tags=self.with_new_prediction_tags)

    def test_givenAddressParser_whenRetrainNewTagsNoEOS_thenTrainingDoesNotOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        with self.assertRaises(ValueError):
            _ = address_parser.retrain(self.new_prediction_data_container,
                                       self.a_train_ratio,
                                       epochs=self.a_single_epoch,
                                       batch_size=self.a_batch_size,
                                       num_workers=self.a_number_of_workers,
                                       logging_path=self.a_checkpoints_saving_dir,
                                       prediction_tags={"ATag": 0})

    def test_givenABPEmbAddressParser_whenRetrainNewTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainMultipleEpochsNewTags_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_three_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithConfigNewTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithConfigWithCallbacksNewTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        callback_mock = MagicMock(spec=Callback)
        performance_after_training = address_parser.retrain(self.new_prediction_data_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            callbacks=[callback_mock],
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.with_new_prediction_tags)

        self.assertIsNotNone(performance_after_training)

        callback_train_start_call = [call.on_train_begin({})]
        callback_mock.assert_has_calls(callback_train_start_call)
        callback_train_end_call = [call.on_train_end({})]
        callback_mock.assert_has_calls(callback_train_end_call)
        callback_mock.assert_not_called()

    # Test API tests
    def test_givenAFasttextAddressParser_whenTestWithNumWorkerAt0NewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_zero_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithNumWorkerGreaterThen0NewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestMultipleEpochsNewTags_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigNewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigWithCallbacksNewTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     callbacks=[callback_mock])

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

    def test_givenAFasttextAddressParser_whenTestWithFasttextCkptNewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithStrCkptNewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNumberWorkersAt0NewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_zero_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNumberWorkersGreaterThen0NewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestMultipleEpochsNewTags_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigNewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigWithCallbacksNewTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     callbacks=[callback_mock])

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

    def test_givenABPEmbAddressParser_whenTestWithBPEmbCkptNewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithStrCkptNewTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        self.training(address_parser,
                      self.new_prediction_data_container,
                      self.a_number_of_workers,
                      prediction_tags=self.with_new_prediction_tags)

        performance_after_test = address_parser.test(self.new_prediction_data_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)


if __name__ == "__main__":
    unittest.main()
