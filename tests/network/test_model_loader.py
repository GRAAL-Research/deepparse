# Since we use patch to mock the weights upload, we skip the unused argument error.
# pylint: disable=unused-argument

import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from deepparse.network import ModelLoader


class ModelLoaderLoadWeightsTest(TestCase):
    def setUp(self) -> None:
        self.a_cache_dir = "a/cache/dir"
        self.a_path = "a/path/to/retrained_model.ckpt"
        self.a_device = "cpu"
        self.a_version = "Finetuned_a_version"
        self.a_state_dict = {"layer.weight": 0}
        # The torch archive always wraps the weights with metadata at the top level.
        self.a_checkpoint = {
            "address_tagger_model": self.a_state_dict,
            "model_type": "fasttext",
            "version": self.a_version,
        }

    @patch("deepparse.network.model_loader.handle_weights_upload")
    def test_givenACheckpoint_whenLoadWeights_thenReturnsTheCheckpointVersion(self, weights_upload_mock):
        # The version lives at the top level of the checkpoint, not inside the state dict. Reading it after
        # extracting the state dict (the previous bug) would always yield None.
        weights_upload_mock.return_value = self.a_checkpoint
        model_mock = MagicMock()

        _, actual_version = ModelLoader(cache_dir=self.a_cache_dir).load_weights(
            model=model_mock, path_to_model_torch_archive=self.a_path, device=self.a_device
        )

        self.assertEqual(actual_version, self.a_version)

    @patch("deepparse.network.model_loader.handle_weights_upload")
    def test_givenACheckpoint_whenLoadWeights_thenLoadsTheInnerStateDictIntoTheModel(self, weights_upload_mock):
        weights_upload_mock.return_value = self.a_checkpoint
        model_mock = MagicMock()

        ModelLoader(cache_dir=self.a_cache_dir).load_weights(
            model=model_mock, path_to_model_torch_archive=self.a_path, device=self.a_device
        )

        model_mock.load_state_dict.assert_called_once_with(self.a_state_dict)


if __name__ == "__main__":
    unittest.main()
