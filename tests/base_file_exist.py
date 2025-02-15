from os.path import exists
from pathlib import Path
from typing import Union
from unittest import TestCase


class FileCreationTestCase(TestCase):
    def assertFileExist(self, file_path: Union[str, Path]) -> None:
        file_exists = exists(file_path)
        self.assertTrue(file_exists)
