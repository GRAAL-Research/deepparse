# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, too-many-public-methods

import os
import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call, patch

from poutyne import Callback

from deepparse.parser import AddressParser
from tests.base_capture_output import CaptureOutputTestCase
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner")
class AddressParserIntegrationRetrainTest(AddressParserRetrainTestCase, CaptureOutputTestCase):

    def test_givenAFasttextAddressParser_whenRetrain_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_training)

    @patch("deepparse.parser.address_parser.poutyne")
    def test_givenAnAddressParser_whenRetrainWithPoutyne17andBefore_thenTrainingOccurWithAWarningPrint(
            self, poutyne_mock):
        poutyne_mock.version.__version__ = 1.7
        self._capture_output()
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        address_parser.retrain(self.training_container,
                               self.a_train_ratio,
                               epochs=self.a_single_epoch,
                               batch_size=self.a_batch_size,
                               num_workers=self.a_number_of_workers,
                               logging_path=self.a_checkpoints_saving_dir)

        actual = self.test_out.getvalue()

        expected = "You are using a older version of Poutyne that does not support properly error management." \
                   " Due to that, we cannot show retrain progress. To fix that, update Poutyne to the newest version.\n"

        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.poutyne")
    def test_givenAnAddressParser_whenRetrainWithPoutyne18andAfter_thenTrainingOccurWithoutAWarningPrint(
            self, poutyne_mock):
        poutyne_mock.version.__version__ = 1.8
        self._capture_output()
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)

        address_parser.retrain(self.training_container,
                               self.a_train_ratio,
                               epochs=self.a_single_epoch,
                               batch_size=self.a_batch_size,
                               num_workers=self.a_number_of_workers,
                               logging_path=self.a_checkpoints_saving_dir)

        actual = self.test_out.getvalue()

        not_expected = "You are using a older version of Poutyne that does not support properly error management." \
                       " Due to that, we cannot show retrain progress. To fix that, update Poutyne to the newest " \
                       "version.\n"

        self.assertNotRegex(actual, not_expected)

    def test_givenAFasttextAddressParser_whenRetrainMultipleEpochs_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_cpu_device,
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
                                       device=self.a_cpu_device,
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
                                       device=self.a_cpu_device,
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
                                       device=self.a_cpu_device,
                                       verbose=self.verbose)
        with self.assertRaises(ValueError):
            address_parser.retrain(self.training_container,
                                   self.a_train_ratio,
                                   epochs=self.a_single_epoch,
                                   batch_size=self.a_batch_size,
                                   num_workers=self.a_number_of_workers,
                                   logging_path=self.a_checkpoints_saving_dir)

    def test_givenABPEmbAddressParser_whenRetrain_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_cpu_device,
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
                                       device=self.a_cpu_device,
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
                                       device=self.a_cpu_device,
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
                                       device=self.a_cpu_device,
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
