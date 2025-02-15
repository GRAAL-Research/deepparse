# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, too-many-public-methods

import os
import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call, ANY

from deepparse.errors import FastTextModelError
from deepparse.parser import AddressParser
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class AddressParserIntegrationTestAPICPUTest(AddressParserRetrainTestCase):
    def test_givenAFasttextAddressParser_whenTestWithNumWorkerAt0_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_zero_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithNumWorkerGreaterThen0_thenTestOccur(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestMultipleEpochs_thenTestOccurCorrectly(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfig_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigWithCallbacks_thenCallbackAreUse(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            callbacks=[callback_mock],
        )

        self.assertIsNotNone(performance_after_test)

        callback_test_start_call = [call.on_test_begin({})]
        callback_mock.assert_has_calls(callback_test_start_call)
        callback_test_end_call = [
            call.on_test_end(
                {
                    "time": ANY,
                    "test_loss": performance_after_test["test_loss"],
                    "test_accuracy": performance_after_test["test_accuracy"],
                }
            )
        ]
        callback_mock.assert_has_calls(callback_test_end_call)
        callback_mock.assert_not_called()

    def test_givenAFasttextAddressParser_whenTestWithFasttextCkpt_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithStrCkpt_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNumberWorkersAt0_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_zero_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNumberWorkersGreaterThen0_thenTestOccur(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestMultipleEpochs_thenTestOccurCorrectly(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfig_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigWithCallbacks_thenCallbackAreUse(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            callbacks=[callback_mock],
        )

        self.assertIsNotNone(performance_after_test)

        callback_test_start_call = [call.on_test_begin({})]
        callback_mock.assert_has_calls(callback_test_start_call)
        callback_test_end_call = [
            call.on_test_end(
                {
                    "time": ANY,
                    "test_loss": performance_after_test["test_loss"],
                    "test_accuracy": performance_after_test["test_accuracy"],
                }
            )
        ]
        callback_mock.assert_has_calls(callback_test_end_call)
        callback_mock.assert_not_called()

    def test_givenABPEmbAddressParser_whenTestWithBPEmbCkpt_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithStrCkpt_thenTestOccur(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextLightAddressParser_whenTest_thenTestDoesNotOccur(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_fasttext_light_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        with self.assertRaises(FastTextModelError):
            address_parser.test(
                self.test_container,
                batch_size=self.a_batch_size,
                num_workers=self.a_number_of_workers,
            )

    def test_givenAFasttextAddressParser_whenTestWithNumWorkerGT0_thenTestDoesNotOccur(
        self,
    ):
        address_parser = AddressParser(
            model_type=self.a_fasttext_light_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        num_worker_gt_0 = 1
        with self.assertRaises(FastTextModelError):
            address_parser.test(
                self.test_container,
                batch_size=self.a_batch_size,
                num_workers=num_worker_gt_0,
            )


if __name__ == "__main__":
    unittest.main()
