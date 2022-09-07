# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import skipIf

import torch

from deepparse import download_fasttext_embeddings, download_fasttext_magnitude_embeddings
from tests.base_file_exist import FileCreationTestCase


@skipIf(not torch.cuda.is_available(), "Cannot run test on GitHub action runner.")
class IntegrationFastTextToolsTests(FileCreationTestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.a_cache_dir = self.temp_dir_obj.name
        self.a_fasttext_file_name_path = os.path.join(self.a_cache_dir, "cc.fr.300.bin")
        self.a_fasttext_light_name_path = os.path.join(self.a_cache_dir, "fasttext.magnitude")

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_integrationDownloadFastTextEmbeddings(self):
        expected = self.a_fasttext_file_name_path
        actual = download_fasttext_embeddings(self.a_cache_dir)
        self.assertEqual(expected, actual)

        self.assertFileExist(actual)

    def test_integrationDownloadFastTextMagnitudeEmbeddings(self):
        expected = self.a_fasttext_light_name_path
        actual = download_fasttext_magnitude_embeddings(self.a_cache_dir)
        self.assertEqual(expected, actual)

        self.assertFileExist(actual)
