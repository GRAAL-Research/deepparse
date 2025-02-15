# pylint: disable=too-many-public-methods
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch, call

from cloudpathlib import S3Path

from deepparse import handle_weights_upload


# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with


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


if __name__ == "__main__":
    unittest.main()
