import logging
import unittest

from deepparse.embeddings_models import EmbeddingsModel


class CallAbstractedEmbeddingsModel(EmbeddingsModel):
    def __call__(self, *args, **kwargs):
        pass


class EmbeddingsModelInterfaceTest(unittest.TestCase):
    def test_whenInstantiated_thenInitProperly(self):
        embeddings_model = CallAbstractedEmbeddingsModel(verbose=False)

        self.assertIsNone(embeddings_model.model)

    def test_whenInstantiatedVerbose_thenVerbose(self):
        with self.assertLogs("deepparse.embeddings_models.embeddings_model", level="INFO") as cm:
            _ = CallAbstractedEmbeddingsModel(verbose=True)

        self.assertTrue(any("Loading the embeddings model" in msg for msg in cm.output))

    def test_whenInstantiatedNotVerbose_thenNoVerbose(self):
        logger = logging.getLogger("deepparse.embeddings_models.embeddings_model")
        with self.assertRaises(AssertionError):
            # assertLogs raises AssertionError if no log is emitted
            with self.assertLogs(logger, level="INFO"):
                _ = CallAbstractedEmbeddingsModel(verbose=False)

    def test_whenInstantiated_thenInitWithoutDim(self):
        embeddings_model = CallAbstractedEmbeddingsModel(verbose=False)
        with self.assertRaises(AttributeError):
            _ = embeddings_model.dim


if __name__ == "__main__":
    unittest.main()
