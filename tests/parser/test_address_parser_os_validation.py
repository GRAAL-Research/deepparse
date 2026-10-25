# pylint: disable=protected-access

import unittest
import warnings
from unittest import TestCase
from unittest.mock import patch

from deepparse.errors import FastTextModelError
from deepparse.parser import AddressParser


class AddressParserOSValidationTest(TestCase):
    @staticmethod
    def an_address_parser(model_type: str) -> AddressParser:
        # We only need the model type to validate the OS, so we skip the models loading of the constructor.
        address_parser = AddressParser.__new__(AddressParser)
        address_parser.model_type = model_type
        return address_parser

    def test_givenAFasttextModelOnWindows_whenValidateWithWorkers_thenDoNotRaise(self):
        address_parser = self.an_address_parser("fasttext")

        with patch("deepparse.parser.address_parser.system", return_value="Windows"):
            address_parser._model_os_validation(num_workers=2)

    def test_givenAFasttextAttentionModelOnWindows_whenValidateWithWorkers_thenDoNotRaise(self):
        address_parser = self.an_address_parser("fasttext-attention")

        with patch("deepparse.parser.address_parser.system", return_value="Windows"):
            address_parser._model_os_validation(num_workers=2)

    def test_givenAFasttextLightModelOnWindows_whenValidateWithWorkers_thenRaise(self):
        address_parser = self.an_address_parser("fasttext-light")

        with patch("deepparse.parser.address_parser.system", return_value="Windows"):
            with self.assertRaises(FastTextModelError):
                address_parser._model_os_validation(num_workers=1)

    def test_givenAFasttextLightModelOnWindows_whenValidateWithoutWorkers_thenDoNotRaise(self):
        address_parser = self.an_address_parser("fasttext-light")

        with patch("deepparse.parser.address_parser.system", return_value="Windows"):
            address_parser._model_os_validation(num_workers=0)

    def test_givenAFasttextModelOnMacOS_whenValidateWithWorkers_thenKeepTheStartMethodAndDoNotWarn(self):
        address_parser = self.an_address_parser("fasttext")

        with (
            patch("deepparse.parser.address_parser.system", return_value="Darwin"),
            patch("deepparse.parser.address_parser.torch.multiprocessing.set_start_method") as set_start_method_mock,
            warnings.catch_warnings(record=True) as caught_warnings,
        ):
            warnings.simplefilter("always")
            address_parser._model_os_validation(num_workers=2)

        set_start_method_mock.assert_not_called()
        self.assertEqual([], [warning for warning in caught_warnings if issubclass(warning.category, UserWarning)])

    def test_givenAFasttextLightModelOnMacOS_whenValidateWithWorkers_thenSetForkAndWarn(self):
        address_parser = self.an_address_parser("fasttext-light")

        with (
            patch("deepparse.parser.address_parser.system", return_value="Darwin"),
            patch("deepparse.parser.address_parser.torch.multiprocessing.set_start_method") as set_start_method_mock,
            warnings.catch_warnings(record=True) as caught_warnings,
        ):
            warnings.simplefilter("always")
            address_parser._model_os_validation(num_workers=2)

        set_start_method_mock.assert_called_once_with("fork")
        self.assertTrue(any(issubclass(warning.category, UserWarning) for warning in caught_warnings))


if __name__ == "__main__":
    unittest.main()
