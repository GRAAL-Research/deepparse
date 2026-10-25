# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf, skipUnless

import numpy as np

from deepparse.download_tools import CACHE_PATH
from deepparse.embeddings_models.fasttext_bin_reader import FastTextBinReader
from tests.fixtures.fasttext_bin_writer import A_TINY_VOCABULARY, create_tiny_fasttext_bin

try:
    from gensim.models.fasttext import FastText, load_facebook_vectors, save_facebook_model
    from gensim.test.utils import common_texts

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from fasttext.FastText import _FastText

    FASTTEXT_AVAILABLE = True
except ImportError:
    _FastText = None
    FASTTEXT_AVAILABLE = False

SOME_ADDRESS_CORPUS = [
    ["350", "rue", "des", "lilas", "québec", "g1l", "1b6"],
    ["2325", "rue", "de", "l'université", "québec", "g1v", "0a6"],
    ["5000", "chemin", "de", "la", "côte-des-neiges", "montréal", "h3t", "1j4"],
    ["12", "boulevard", "saint-jean-sur-richelieu", "lévis", "canada"],
]

# Out-of-vocabulary words with accents, digits, hyphens, non-BMP characters and the empty string.
SOME_OOV_WORDS = [
    "",
    "a",
    "ab",
    "é",
    "québecois",
    "montreal",
    "trois-rivières",
    "h2x3y7",
    "1234-a",
    "sainte-anne-de-la-pérade",
    "çàèùâêîôûëïü",
    "straße",
    "𝄞clé",
    "abcdefghijklmnopqrstuvwxyz",
]

A_TOLERANCE = 1e-6


class ReaderEquivalenceTestCase(TestCase):
    def assertSameVectors(self, reader, reference, words):
        for word in words:
            with self.subTest(word=word):
                np.testing.assert_allclose(reference(word), reader[word], atol=A_TOLERANCE)


@skipUnless(GENSIM_AVAILABLE, "gensim is not installed")
class FastTextBinReaderGensimEquivalenceTest(ReaderEquivalenceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir_obj = TemporaryDirectory()

        cls.a_tiny_model_path = os.path.join(cls.temp_dir_obj.name, "tiny.bin")
        create_tiny_fasttext_bin(cls.a_tiny_model_path)

        # A slightly larger model trained by gensim and saved in the Facebook format.
        cls.a_trained_model_path = os.path.join(cls.temp_dir_obj.name, "trained.bin")
        corpus = common_texts + SOME_ADDRESS_CORPUS * 3
        model = FastText(vector_size=8, window=3, min_count=1, bucket=2000, seed=42, workers=1)
        model.build_vocab(corpus_iterable=corpus)
        model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=2)
        save_facebook_model(model, cls.a_trained_model_path)
        cls.a_trained_vocabulary = list(model.wv.index_to_key)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def test_givenTheWriterFixture_whenCompareToGensim_thenSameInVocabularyVectors(self):
        reader = FastTextBinReader(self.a_tiny_model_path)
        gensim_vectors = load_facebook_vectors(self.a_tiny_model_path)

        # gensim computes the n-grams of "</s>" while fastText does not, so we leave it out (the reader follows
        # fastText, see the fastText equivalence test).
        self.assertSameVectors(reader, gensim_vectors.get_vector, [w for w in A_TINY_VOCABULARY if w != "</s>"])

    def test_givenTheWriterFixture_whenCompareToGensim_thenSameOutOfVocabularyVectors(self):
        reader = FastTextBinReader(self.a_tiny_model_path)
        gensim_vectors = load_facebook_vectors(self.a_tiny_model_path)

        self.assertSameVectors(reader, gensim_vectors.get_vector, SOME_OOV_WORDS)

    def test_givenAGensimTrainedModel_whenCompareToGensim_thenSameInVocabularyVectors(self):
        reader = FastTextBinReader(self.a_trained_model_path)
        gensim_vectors = load_facebook_vectors(self.a_trained_model_path)

        self.assertSameVectors(reader, gensim_vectors.get_vector, self.a_trained_vocabulary)

    def test_givenAGensimTrainedModel_whenCompareToGensim_thenSameOutOfVocabularyVectors(self):
        reader = FastTextBinReader(self.a_trained_model_path)
        gensim_vectors = load_facebook_vectors(self.a_trained_model_path)

        self.assertSameVectors(reader, gensim_vectors.get_vector, SOME_OOV_WORDS)


@skipUnless(FASTTEXT_AVAILABLE, "fasttext is not installed")
class FastTextBinReaderFastTextEquivalenceTest(ReaderEquivalenceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_tiny_model_path = os.path.join(cls.temp_dir_obj.name, "tiny.bin")
        create_tiny_fasttext_bin(cls.a_tiny_model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def test_givenTheWriterFixture_whenCompareToFastText_thenSameVectors(self):
        reader = FastTextBinReader(self.a_tiny_model_path)
        fasttext_model = _FastText(model_path=self.a_tiny_model_path)

        self.assertSameVectors(reader, fasttext_model.get_word_vector, A_TINY_VOCABULARY + SOME_OOV_WORDS)


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
@skipUnless(GENSIM_AVAILABLE, "gensim is not installed")
@skipUnless(os.path.isfile(os.path.join(CACHE_PATH, "cc.fr.300.bin")), "cc.fr.300.bin is not in the cache")
class FastTextBinReaderRealEmbeddingsEquivalenceTest(ReaderEquivalenceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # We do not download the embeddings (~7 GO); the test only runs if they are already in the cache.
        cls.a_model_path = os.path.join(CACHE_PATH, "cc.fr.300.bin")
        cls.reader = FastTextBinReader(cls.a_model_path)
        cls.gensim_vectors = load_facebook_vectors(cls.a_model_path)

    def test_givenTheFrenchEmbeddings_whenCompareToGensim_thenSameInVocabularyVectors(self):
        # We skip the first word, "</s>", whose subwords gensim computes but fastText does not.
        some_words = self.gensim_vectors.index_to_key[1:2001]

        self.assertSameVectors(self.reader, self.gensim_vectors.get_vector, some_words)

    def test_givenTheFrenchEmbeddings_whenCompareToGensim_thenSameAddressAndOutOfVocabularyVectors(self):
        some_words = SOME_OOV_WORDS + [word for address in SOME_ADDRESS_CORPUS for word in address]

        self.assertSameVectors(self.reader, self.gensim_vectors.get_vector, some_words)
