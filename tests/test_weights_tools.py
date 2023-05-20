# pylint: disable=too-many-public-methods
import os

# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with


import unittest
from unittest import TestCase, skipIf
from unittest.mock import MagicMock, patch, call

from cloudpathlib import S3Path, CloudPath

from deepparse import handle_weights_upload
from deepparse.parser import AddressParser, FormattedParsedAddress


class WeightsToolsTests(TestCase):
    @patch("deepparse.weights_tools.torch")
    def test_givenAS3Path_whenHandleWeights_upload_thenReturnProperWeights(self, torch_mock):
        s3_path = MagicMock(spec=S3Path)

        weights_mock = MagicMock()
        torch_mock.load().return_value = weights_mock

        handle_weights_upload(path_to_model_to_upload=s3_path)

        torch_mock.has_calls([call.load()])

    @patch("deepparse.weights_tools.CloudPath")
    @patch("deepparse.weights_tools.torch")
    def test_givenAStringS3Path_whenHandleWeights_upload_thenReturnProperWeights(self, torch_mock, cloud_path_mock):
        s3_path = "s3://a_path"

        weights_mock = MagicMock()
        torch_mock.load().return_value = weights_mock

        handle_weights_upload(path_to_model_to_upload=s3_path)

        torch_mock.has_calls([call.load()])
        cloud_path_mock.assert_called()

    @patch("deepparse.weights_tools.CloudPath")
    @patch("deepparse.weights_tools.torch")
    def test_givenAStringPath_whenHandleWeights_upload_thenReturnProperWeights(self, torch_mock, cloud_path_mock):
        s3_path = "a_normal_path.ckpt"

        weights_mock = MagicMock()
        torch_mock.load().return_value = weights_mock

        handle_weights_upload(path_to_model_to_upload=s3_path)

        torch_mock.has_calls([call.load()])

        cloud_path_mock.assert_not_called()

    def test_givenAWrongfullyStringS3Path_whenHandleWeights_upload_thenRaiseError(self):
        s3_path = "s3/model.ckpt"

        with self.assertRaises(FileNotFoundError):
            handle_weights_upload(path_to_model_to_upload=s3_path)

        s3_path = "s3//model.ckpt"

        with self.assertRaises(FileNotFoundError):
            handle_weights_upload(path_to_model_to_upload=s3_path)

    @skipIf(
        not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
        "download of model too long for test in runner",
    )
    def test_integration_handle_weights_with_uri(self):
        uri = "s3://deepparse/fasttext.ckpt"

        address_parser = AddressParser(model_type="fasttext", path_to_retrained_model=uri)
        parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
        self.assertIsInstance(parse_address, FormattedParsedAddress)

        uri = CloudPath("s3://deepparse/fasttext.ckpt")

        address_parser = AddressParser(model_type="fasttext", path_to_retrained_model=uri)
        parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
        self.assertIsInstance(parse_address, FormattedParsedAddress)


if __name__ == "__main__":
    unittest.main()
