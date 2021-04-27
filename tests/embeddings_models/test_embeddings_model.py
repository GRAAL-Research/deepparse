import io
import sys
import unittest
from unittest import TestCase

from deepparse.embeddings_models import EmbeddingsModel


class CallAbstractedEmbeddingsModel(EmbeddingsModel):

    def __call__(self, *args, **kwargs):
        pass


class EmbeddingsModelInterfaceTest(TestCase):

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out

    def test_whenInstantiated_thenInitProperly(self):
        embeddings_model = CallAbstractedEmbeddingsModel(verbose=False)

        self.assertIsNone(embeddings_model.model)

    def test_whenInstantiatedVerbose_thenVerbose(self):
        self._capture_output()
        _ = CallAbstractedEmbeddingsModel(verbose=True)

        actual = self.test_out.getvalue().strip()
        expected = "Loading the embeddings model"
        self.assertEqual(actual, expected)

    def test_whenInstantiatedNotVerbose_thenNoVerbose(self):
        self._capture_output()
        _ = CallAbstractedEmbeddingsModel(verbose=False)

        actual = self.test_out.getvalue().strip()
        expected = ""
        self.assertEqual(actual, expected)

    def test_whenInstantiated_thenInitWithoutDim(self):
        embeddings_model = CallAbstractedEmbeddingsModel(verbose=False)
        with self.assertRaises(AttributeError):
            _ = embeddings_model.dim


if __name__ == "__main__":
    unittest.main()
