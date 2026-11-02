"""
Freeze the public API of the deepparse top-level package. If one of these tests fails, the public API changed: make
sure the change is deliberate (a removal needs a deprecation cycle until the next major version, see the
"Versioning" section of the documentation) and update the expected lists below.
"""

import os
import subprocess
import sys
import unittest
import warnings
from unittest import TestCase

import deepparse

EXPECTED_PUBLIC_API = [
    "BPEmbLoader",
    "CACHE_PATH",
    "MODEL_MAPPING_CHOICES",
    "download_fasttext_embeddings",
    "download_fasttext_magnitude_embeddings",
    "download_model",
    "download_model2vec_embeddings",
    "download_models",
    "download_weights",
    "download_worldwide_addresses_sample",
    "load_version",
]

# The deprecated APIs and the internal helpers and constants that were reachable as deepparse.<name> in 0.11.0 (before
# the public API was delimited with __all__) and were removed in 1.0.0. Import the helpers from their own module.
NAMES_REMOVED_IN_1_0_0 = [
    "BPEmbBaseURLWrapperBugFix",
    "DataError",
    "FASTTEXT_AVAILABLE",
    "FASTTEXT_COPYRIGHT_MIT_LICENSE",
    "HTTP_CLIENT_ERROR_STATUS_CODE",
    "MODEL_REPO_IDS",
    "NEXT_RANGE_STATUS_CODE",
    "PYARROW_AVAILABLE",
    "WORLDWIDE_ADDRESSES_REMOVED_TAGS",
    "WORLDWIDE_ADDRESSES_REPO_ID",
    "WORLDWIDE_ADDRESSES_TAGS_MAPPING",
    "bpemb_url_bug_fix",
    "download_gz_model",
    "extract_package_version",
    "handle_weights_upload",
    "load_fasttext_embeddings",
    "load_word2vec_binary_vectors",
    "valid_poutyne_version",
    "validate_data_to_parse",
    "validate_if_any_empty",
    "validate_if_any_none",
    "validate_if_any_whitespace_only",
    "weights_init",
]

# The submodules of deepparse stay reachable as deepparse.<name>.
SUBMODULES = ["bpemb_loader", "data_validation", "download_tools", "errors", "validations", "version", "weights_tools"]

# Standard library and third-party modules and objects that 0.11.0 re-exported by accident through the wildcard
# imports. They were never part of the API and are no longer reachable from deepparse.
ACCIDENTAL_REEXPORTS_DROPPED_IN_0_12_0 = [
    "CloudPath",
    "Dict",
    "HfFileSystem",
    "List",
    "OrderedDict",
    "Path",
    "S3Path",
    "Tuple",
    "Union",
    "cached_file",
    "disable_progress_bar",
    "enable_progress_bar",
    "extract_commit_hash",
    "gzip",
    "hf_hub_download",
    "importlib",
    "init",
    "logger",
    "logging",
    "nn",
    "np",
    "os",
    "poutyne",
    "pq",
    "re",
    "requests",
    "sentencepiece",
    "shutil",
    "snapshot_download",
    "sys",
    "tarfile",
    "tempfile",
    "torch",
    "urlopen",
    "warnings",
]


class PublicAPITest(TestCase):
    def test_theTopLevelAllIsTheExpectedPublicAPI(self):
        self.assertEqual(sorted(EXPECTED_PUBLIC_API), sorted(deepparse.__all__))
        self.assertEqual(len(deepparse.__all__), len(set(deepparse.__all__)))

    def test_everyNameOfTheAllIsAnAttribute(self):
        for name in deepparse.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(deepparse, name))

    def test_theNamesRemovedIn100AreNoLongerAttributes(self):
        for name in NAMES_REMOVED_IN_1_0_0:
            with self.subTest(name=name):
                self.assertFalse(hasattr(deepparse, name))

    def test_theSubmodulesAreStillAttributes(self):
        for name in SUBMODULES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(deepparse, name))

    def test_theAccidentalReexportsOf0110AreNoLongerAttributes(self):
        for name in ACCIDENTAL_REEXPORTS_DROPPED_IN_0_12_0:
            with self.subTest(name=name):
                self.assertFalse(hasattr(deepparse, name))

    def test_whenWildcardImport_thenImportExactlyTheAll(self):
        namespace: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            exec("from deepparse import *", namespace)  # pylint: disable=exec-used

        self.assertEqual(sorted(EXPECTED_PUBLIC_API), sorted(name for name in namespace if name != "__builtins__"))

    def test_whenImportDeepparse_thenDoNotEmitADeprecationWarning(self):
        # In a fresh interpreter, since deepparse is already imported (and cached) in this one.
        completed_process = subprocess.run(
            [sys.executable, "-W", "error::DeprecationWarning:deepparse", "-c", "import deepparse"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, completed_process.returncode, completed_process.stderr)


if __name__ == "__main__":
    os.environ.setdefault("TEST_LEVEL", "unit")
    unittest.main()
