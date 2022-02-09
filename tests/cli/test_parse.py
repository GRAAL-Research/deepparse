# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument

import argparse
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from deepparse.cli import download


class ParseTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.fake_cache_path = os.path.join(self.temp_dir_obj.name, "fake_cache")
        self.a_fasttext_model_type = "fasttext"
        self.a_fasttext_att_model_type = "fasttext_attention"
        self.a_fasttext_light_model_type = "fasttext-light"
        self.a_bpemb_model_type = "bpemb"
        self.a_bpemb_att_model_type = "bpemd_attention"

        self.create_parser()

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "parsing_model",
            choices=[
                self.a_fasttext_model_type,
                self.a_fasttext_att_model_type,
                self.a_fasttext_light_model_type,
                self.a_bpemb_model_type,
                self.a_bpemb_att_model_type,
            ],
        )

        self.parser.add_argument("dataset_path", type=str)

        self.parser.add_argument("export_file_name", type=str)

        self.parser.add_argument("--device", type=str, default="0")

        self.parser.add_argument("--path_to_retrained_model", type=str, default=None)

        self.parser.add_argument("--csv_column_name", type=str, default=None)

        self.parser.add_argument("--csv_column_separator", type=str, default="\t")


if __name__ == "__main__":
    unittest.main()
