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
    "download_models",
    "download_weights",
    "download_worldwide_addresses_sample",
    "load_version",
    # Deprecated, will be removed in 1.0.0.
    "BPEmbBaseURLWrapperBugFix",
    "FASTTEXT_AVAILABLE",
    "load_fasttext_embeddings",
]

# Every deepparse name (helpers, constants and submodules) reachable as deepparse.<name> in 0.11.0, before the public
# API was delimited with __all__. They must all stay reachable until 1.0.0, even the ones that are not in __all__.
NAMES_REACHABLE_IN_0_11_0 = [
    "BPEmbBaseURLWrapperBugFix",
    "CACHE_PATH",
    "DataError",
    "FASTTEXT_AVAILABLE",
    "FASTTEXT_COPYRIGHT_MIT_LICENSE",
    "HTTP_CLIENT_ERROR_STATUS_CODE",
    "MODEL_MAPPING_CHOICES",
    "MODEL_REPO_IDS",
    "NEXT_RANGE_STATUS_CODE",
    "PYARROW_AVAILABLE",
    "WORLDWIDE_ADDRESSES_REMOVED_TAGS",
    "WORLDWIDE_ADDRESSES_REPO_ID",
    "WORLDWIDE_ADDRESSES_TAGS_MAPPING",
    "bpemb_url_bug_fix",
    "data_validation",
    "download_fasttext_embeddings",
    "download_fasttext_magnitude_embeddings",
    "download_gz_model",
    "download_model",
    "download_models",
    "download_tools",
    "download_weights",
    "download_worldwide_addresses_sample",
    "errors",
    "extract_package_version",
    "handle_weights_upload",
    "load_fasttext_embeddings",
    "load_version",
    "load_word2vec_binary_vectors",
    "valid_poutyne_version",
    "validate_data_to_parse",
    "validate_if_any_empty",
    "validate_if_any_none",
    "validate_if_any_whitespace_only",
    "validations",
    "version",
    "weights_init",
    "weights_tools",
]

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

    def test_everyNameReachableIn0110IsStillAnAttribute(self):
        for name in NAMES_REACHABLE_IN_0_11_0:
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
