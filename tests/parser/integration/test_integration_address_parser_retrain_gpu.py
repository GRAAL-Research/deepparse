# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods

import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call

import torch
from poutyne import Callback

from deepparse.parser import AddressParser
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserIntegrationRetrainTest(AddressParserRetrainTestCase):

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


if __name__ == "__main__":
    unittest.main()
