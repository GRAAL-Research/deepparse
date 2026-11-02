# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import os
import pickle
import struct
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepparse.embeddings_models.fasttext_bin_reader import (
    FastTextBinReader,
    compute_char_ngrams,
    fasttext_hash,
)
from tests.fixtures.fasttext_bin_writer import A_TINY_VOCABULARY, create_tiny_fasttext_bin


def unsigned_fnv1a_hash(token: bytes) -> int:
    # The textbook FNV-1a 32-bit hash (bytes are NOT sign-extended), which fastText does not use.
    h = 2166136261
    for byte in token:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


class EmbedWords:
    # A picklable collate_fn for the spawned DataLoader workers.
    def __init__(self, reader: FastTextBinReader) -> None:
        self.reader = reader

    def __call__(self, batch):
        return [torch.tensor(np.array(self.reader[word])) for word in batch]


class FastTextHashTest(TestCase):
    def test_givenAnEmptyToken_whenHash_thenReturnFnvOffsetBasis(self):
        self.assertEqual(2166136261, fasttext_hash(b""))

    def test_givenAnAsciiToken_whenHash_thenReturnFnv1aHash(self):
        # Known FNV-1a 32-bit values; for ASCII bytes, the sign extension changes nothing.
        self.assertEqual(0xE40C292C, fasttext_hash(b"a"))
        self.assertEqual(unsigned_fnv1a_hash(b"rue"), fasttext_hash(b"rue"))
        self.assertEqual(567099551, fasttext_hash(b"rue"))

    def test_givenAnAccentedToken_whenHash_thenSignExtendTheBytes(self):
        # Values validated against the fastText C++ implementation.
        self.assertEqual(1023043777, fasttext_hash("é".encode("utf-8")))
        self.assertEqual(1292435955, fasttext_hash("québec".encode("utf-8")))
        self.assertEqual(2147392514, fasttext_hash("montréal".encode("utf-8")))

    def test_givenAnAccentedToken_whenHash_thenDifferFromUnsignedFnv1a(self):
        a_token = "québec".encode("utf-8")

        self.assertNotEqual(unsigned_fnv1a_hash(a_token), fasttext_hash(a_token))

    def test_givenAnyToken_whenHash_thenFitIn32Bits(self):
        for a_token in ("", "rue", "côte-des-neiges", "𝄞"):
            actual = fasttext_hash(a_token.encode("utf-8"))

            self.assertGreaterEqual(actual, 0)
            self.assertLess(actual, 2**32)


class ComputeCharNgramsTest(TestCase):
    def test_givenAWord_whenComputeNgrams_thenReturnNgramsWithMarkers(self):
        expected = [b"<ru", b"<rue", b"<rue>", b"rue", b"rue>", b"ue>"]

        actual = compute_char_ngrams(b"<rue>", 3, 6)

        self.assertEqual(expected, actual)

    def test_givenAShortWord_whenComputeNgrams_thenKeepTheWholeWrappedWord(self):
        # fastText keeps "<ab>" since it has 4 characters, which is within [minn, maxn].
        expected = [b"<ab", b"<ab>", b"ab>"]

        actual = compute_char_ngrams(b"<ab>", 3, 6)

        self.assertEqual(expected, actual)

    def test_givenALongWord_whenComputeNgrams_thenNgramsAreAtMostMaxnLong(self):
        actual = compute_char_ngrams(b"<abcdefgh>", 3, 6)

        # "<abcdefgh>" has 10 characters, so 8 + 7 + 6 + 5 n-grams of length 3 to 6.
        self.assertEqual(8 + 7 + 6 + 5, len(actual))
        self.assertTrue(all(3 <= len(ngram) <= 6 for ngram in actual))

    def test_givenAnAccentedWord_whenComputeNgrams_thenCountCharactersNotBytes(self):
        expected = ["<ét", "<été", "<été>", "été", "été>", "té>"]

        actual = compute_char_ngrams("<été>".encode("utf-8"), 3, 6)

        self.assertEqual(expected, [ngram.decode("utf-8") for ngram in actual])

    def test_givenMinnOfOne_whenComputeNgrams_thenSkipTheLoneMarkers(self):
        expected = [b"<a", b"a", b"a>"]

        actual = compute_char_ngrams(b"<a>", 1, 2)

        self.assertEqual(expected, actual)

    def test_givenAWordShorterThanMinn_whenComputeNgrams_thenReturnNoNgram(self):
        self.assertEqual([], compute_char_ngrams(b"<>", 3, 6))


class FastTextBinReaderTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir_obj = TemporaryDirectory()
        cls.a_model_path = os.path.join(cls.temp_dir_obj.name, "tiny.bin")
        cls.an_input_matrix = create_tiny_fasttext_bin(cls.a_model_path)
        cls.a_dim = 4
        cls.a_bucket = 1000
        cls.nwords = len(A_TINY_VOCABULARY)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def setUp(self) -> None:
        self.reader = FastTextBinReader(self.a_model_path)

    def bucket_row(self, ngram: str) -> np.ndarray:
        return self.an_input_matrix[self.nwords + fasttext_hash(ngram.encode("utf-8")) % self.a_bucket]

    def test_givenATinyModel_whenRead_thenParseTheArgs(self):
        self.assertEqual(self.a_dim, self.reader.dim)
        self.assertEqual(self.a_dim, self.reader.get_dimension())
        self.assertEqual(self.a_bucket, self.reader.bucket)
        self.assertEqual(3, self.reader.minn)
        self.assertEqual(6, self.reader.maxn)
        self.assertEqual(1, self.reader.args["model"])
        self.assertEqual(2, self.reader.args["loss"])
        self.assertAlmostEqual(1e-4, self.reader.args["t"])

    def test_givenATinyModel_whenRead_thenParseTheDictionary(self):
        self.assertEqual(self.nwords, self.reader.nwords)
        self.assertEqual(self.nwords, self.reader.ntokens)
        for expected_id, a_word in enumerate(A_TINY_VOCABULARY):
            self.assertEqual(expected_id, self.reader.get_word_id(a_word))
        self.assertEqual(-1, self.reader.get_word_id("boulevard"))

    def test_givenATinyModel_whenRead_thenMemoryMapTheInputMatrix(self):
        # pylint: disable=protected-access
        self.assertIsInstance(self.reader._input_matrix, np.memmap)
        np.testing.assert_array_equal(self.an_input_matrix, self.reader._input_matrix)

    def test_givenAnInVocabularyWord_whenGetSubwordIds_thenWordIdFirstThenNgramBuckets(self):
        expected = [A_TINY_VOCABULARY.index("rue")] + [
            self.nwords + fasttext_hash(ngram.encode("utf-8")) % self.a_bucket
            for ngram in ("<ru", "<rue", "<rue>", "rue", "rue>", "ue>")
        ]

        actual = self.reader.get_subword_ids("rue")

        self.assertEqual(expected, actual)

    def test_givenAnInVocabularyWord_whenGetItem_thenReturnMeanOfWordAndNgramRows(self):
        rows = [self.an_input_matrix[A_TINY_VOCABULARY.index("rue")]]
        rows += [self.bucket_row(ngram) for ngram in ("<ru", "<rue", "<rue>", "rue", "rue>", "ue>")]
        expected = np.mean(rows, axis=0)

        actual = self.reader["rue"]

        np.testing.assert_allclose(expected, actual, atol=1e-6)
        self.assertEqual(np.float32, actual.dtype)
        self.assertEqual((self.a_dim,), actual.shape)

    def test_givenAnAccentedInVocabularyWord_whenGetItem_thenReturnMeanOfWordAndNgramRows(self):
        ngrams = ["<qu", "<qué", "<québ", "<québe", "qué", "québ", "québe", "québec", "uéb", "uébe", "uébec"]
        ngrams += ["uébec>", "ébe", "ébec", "ébec>", "bec", "bec>", "ec>"]
        rows = [self.an_input_matrix[A_TINY_VOCABULARY.index("québec")]] + [self.bucket_row(n) for n in ngrams]
        expected = np.mean(rows, axis=0)

        actual = self.reader["québec"]

        np.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_givenAnOutOfVocabularyWord_whenGetItem_thenReturnMeanOfNgramRows(self):
        expected = np.mean([self.bucket_row(ngram) for ngram in ("<ru", "<rus", "<rus>", "rus", "rus>", "us>")], axis=0)

        actual = self.reader["rus"]

        np.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_givenAnOutOfVocabularyWordWithoutNgram_whenGetItem_thenReturnZeros(self):
        # "<>" has only 2 characters, so the empty string has no n-gram with minn 3.
        self.assertEqual([], self.reader.get_subword_ids(""))
        np.testing.assert_array_equal(np.zeros(self.a_dim, dtype=np.float32), self.reader[""])

    def test_givenTheEndOfSentenceToken_whenGetItem_thenReturnOnlyItsWordRow(self):
        # fastText does not compute the subwords of "</s>".
        np.testing.assert_allclose(self.an_input_matrix[0], self.reader["</s>"], atol=1e-6)

    def test_givenAReader_whenPickled_thenReopenTheMemoryMapAndKeepTheVectors(self):
        a_pickled_reader = pickle.dumps(self.reader)

        actual = pickle.loads(a_pickled_reader)

        # The matrix is not part of the pickle; only the path and the dictionary are.
        self.assertLess(len(a_pickled_reader), self.an_input_matrix.nbytes)
        self.assertIsInstance(actual._input_matrix, np.memmap)  # pylint: disable=protected-access
        self.assertEqual(self.reader.path, actual.path)
        for a_word in ("rue", "québec", "boulevard", ""):
            np.testing.assert_array_equal(self.reader[a_word], actual[a_word])

    def test_givenADataLoaderWithSpawnedWorkers_whenIterate_thenReturnTheReaderVectors(self):
        some_words = ["rue", "avenue", "québec", "montréal", "boulevard", "saint-jean", "h3t1j4", "123"]
        data_loader = DataLoader(
            some_words,
            batch_size=2,
            num_workers=2,
            collate_fn=EmbedWords(self.reader),
            multiprocessing_context="spawn",
        )

        actual = [vector for batch in data_loader for vector in batch]

        self.assertEqual(len(some_words), len(actual))
        for a_word, a_vector in zip(some_words, actual):
            np.testing.assert_array_equal(self.reader[a_word], a_vector.numpy())


class FastTextBinReaderValidationTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.a_model_path = os.path.join(self.temp_dir_obj.name, "model.bin")

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def overwrite(self, offset: int, data: bytes) -> None:
        with open(self.a_model_path, "r+b") as file:
            file.seek(offset)
            file.write(data)

    def test_givenAFileWithAWrongMagic_whenRead_thenRaiseValueError(self):
        create_tiny_fasttext_bin(self.a_model_path)
        self.overwrite(0, struct.pack("<i", 300))

        with self.assertRaises(ValueError):
            FastTextBinReader(self.a_model_path)

    def test_givenAFileWithANewerVersion_whenRead_thenRaiseValueError(self):
        create_tiny_fasttext_bin(self.a_model_path)
        self.overwrite(4, struct.pack("<i", 13))

        with self.assertRaises(ValueError):
            FastTextBinReader(self.a_model_path)

    def test_givenAQuantizedModel_whenRead_thenRaiseNotImplementedError(self):
        create_tiny_fasttext_bin(self.a_model_path)
        # The quant_input flag follows the header (8 bytes), the args (56 bytes) and the dictionary.
        dictionary_size = 12 + 16 + sum(len(word.encode("utf-8")) + 1 + 9 for word in A_TINY_VOCABULARY)
        self.overwrite(8 + 56 + dictionary_size, struct.pack("<?", True))

        with self.assertRaises(NotImplementedError):
            FastTextBinReader(self.a_model_path)

    def test_givenAMatrixNotMatchingTheArgs_whenRead_thenRaiseValueError(self):
        create_tiny_fasttext_bin(self.a_model_path)
        dictionary_size = 12 + 16 + sum(len(word.encode("utf-8")) + 1 + 9 for word in A_TINY_VOCABULARY)
        self.overwrite(8 + 56 + dictionary_size + 1, struct.pack("<q", 3))

        with self.assertRaises(ValueError):
            FastTextBinReader(self.a_model_path)

    def test_givenSmallerNgramsBounds_whenGetSubwordIds_thenUseTheModelBounds(self):
        create_tiny_fasttext_bin(self.a_model_path, minn=2, maxn=3, bucket=50)
        reader = FastTextBinReader(self.a_model_path)
        expected = [A_TINY_VOCABULARY.index("rue")] + [
            len(A_TINY_VOCABULARY) + fasttext_hash(ngram) % 50
            for ngram in (b"<r", b"<ru", b"ru", b"rue", b"ue", b"ue>", b"e>")
        ]

        actual = reader.get_subword_ids("rue")

        self.assertEqual(expected, actual)

    def test_givenAMaxnOfZero_whenGetSubwordIds_thenReturnOnlyTheWordId(self):
        create_tiny_fasttext_bin(self.a_model_path, minn=0, maxn=0)
        reader = FastTextBinReader(self.a_model_path)

        self.assertEqual([A_TINY_VOCABULARY.index("rue")], reader.get_subword_ids("rue"))
        self.assertEqual([], reader.get_subword_ids("boulevard"))
