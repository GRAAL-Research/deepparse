# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable, too-many-public-methods

import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call, ANY

import torch

from deepparse.parser import AddressParser
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserIntegrationTestAPITest(AddressParserRetrainTestCase):

    def test_givenAFasttextAddressParser_whenTestWithNumWorkerAt0_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, self.training_container, self.a_zero_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithNumWorkerGreaterThen0_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestMultipleEpochs_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfig_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigWithCallbacks_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.test_container,
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

    def test_givenAFasttextAddressParser_whenTestWithFasttextCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithStrCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNumWorkerAt0_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, self.training_container, self.a_zero_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNumWorkerGreaterThen0_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestMultipleEpochs_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfig_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigWithCallbacks_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.test_container,
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

    def test_givenABPEmbAddressParser_whenTestWithBPEmbCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithStrCkpt_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, self.training_container, self.a_number_of_workers)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers)

        self.assertIsNotNone(performance_after_test)


if __name__ == "__main__":
    unittest.main()
