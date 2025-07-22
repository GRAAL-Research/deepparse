import os
import unittest
from typing import List
from unittest import skipIf

from deepparse.parser import AddressParser
from tests.base_capture_output import CaptureOutputTestCase
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
class AddressParserIntegrationTestAPITest(AddressParserRetrainTestCase, CaptureOutputTestCase):
    def test_givenAnVerboseTrueAddressParser_whenTestOverrideTrue_thenPrint(self):
        self._capture_output()

        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=True,
        )

        performance_after_test = address_parser.test(
            self.test_container, batch_size=self.a_batch_size, verbose=True  # Print
        )

        actual = self.test_out.getvalue().strip()

        expected_contains = [
            "test_loss:",
            str(round(performance_after_test.get("test_loss"), 6)),
            "test_accuracy",
            str(round(performance_after_test.get("test_accuracy"), 6)),
        ]

        self.assertContains(actual, expected_contains)

    def test_givenAnVerboseTrueAddressParser_whenTestOverrideFalse_thenDontPrint(self):
        self._capture_output()

        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=True,
        )

        performance_after_test = address_parser.test(
            self.test_container, batch_size=self.a_batch_size, verbose=False  # No print
        )

        actual = self.test_out.getvalue().strip()

        expected_contains = [
            "test_loss:",
            str(round(performance_after_test.get("test_loss"), 6)),
            "test_accuracy",
            str(round(performance_after_test.get("test_accuracy"), 6)),
        ]

        self.assertNotContains(actual, expected_contains)

    def assertContains(self, actual: str, expected_contains: List):
        for expected_contain in expected_contains:
            self.assertIn(expected_contain, actual)

    def assertNotContains(self, actual: str, not_expected_contains: List):
        for not_expected_contain in not_expected_contains:
            self.assertNotIn(not_expected_contain, actual)


if __name__ == "__main__":
    unittest.main()
