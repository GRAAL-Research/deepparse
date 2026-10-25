# Pylint error for TemporaryDirectory ask for with statement
# pylint: disable=consider-using-with

import io
import os
import pickle
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import sentencepiece

from deepparse.bpemb_url_bug_fix import BPEmbBaseURLWrapperBugFix, load_word2vec_binary_vectors

A_CORPUS = [
    "350 rue des lilas ouest quebec city quebec g0l 0b0",
    "2325 rue de l'universite quebec qc g0v 0a0",
    "1600 pennsylvania avenue nw washington dc 00000",
    "10 downing street london sw0a 0aa united kingdom",
    "221b baker street london nw0 0xe",
] * 20


def write_word2vec_binary(path: Path, words: list, vectors: np.ndarray, newline_after_vector: bool) -> None:
    with open(path, "wb") as file:
        file.write(f"{len(words)} {vectors.shape[1]}\n".encode())
        for word, vector in zip(words, vectors):
            file.write(word.encode() + b" " + vector.astype("<f4").tobytes())
            if newline_after_vector:
                file.write(b"\n")


def a_response(payload: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.raw = io.BytesIO(payload)
    return response


class LoadWord2VecBinaryVectorsTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir_obj = TemporaryDirectory()
        self.a_path = Path(self.temp_dir_obj.name) / "vectors.bin"
        # Words containing a byte equal to a newline or a space in a float would break a naive parser.
        self.words = ["▁a", "b", "▁cé"]
        self.vectors = np.array([[1.0, -2.5, 10.0], [0.0, 1e-8, 3.25], [-7.0, 2.0, 0.5]], dtype=np.float32)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_givenAFileWithoutNewlineAfterVectors_whenLoad_thenReturnTheVectorsInFileOrder(self):
        write_word2vec_binary(self.a_path, self.words, self.vectors, newline_after_vector=False)

        actual = load_word2vec_binary_vectors(self.a_path)

        np.testing.assert_array_equal(self.vectors, actual)
        self.assertEqual(np.float32, actual.dtype)

    def test_givenAFileWithNewlineAfterVectors_whenLoad_thenSkipTheNewlines(self):
        write_word2vec_binary(self.a_path, self.words, self.vectors, newline_after_vector=True)

        actual = load_word2vec_binary_vectors(self.a_path)

        np.testing.assert_array_equal(self.vectors, actual)


class BPEmbBaseURLWrapperBugFixTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = TemporaryDirectory()
        source_dir = Path(cls.temp_dir_obj.name) / "source"
        source_dir.mkdir()

        model_prefix = source_dir / "multi.wiki.bpe.vs40"
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(A_CORPUS),
            model_prefix=str(model_prefix),
            vocab_size=40,
            model_type="bpe",
            minloglevel=2,
        )
        cls.model_bytes = Path(f"{model_prefix}.model").read_bytes()
        cls.spm = sentencepiece.SentencePieceProcessor(model_file=str(Path(f"{model_prefix}.model")))

        cls.vs = 40
        cls.dim = 4
        rng = np.random.default_rng(42)
        cls.vectors = rng.standard_normal((cls.spm.get_piece_size(), cls.dim)).astype(np.float32)
        words = [cls.spm.id_to_piece(index) for index in range(cls.spm.get_piece_size())]
        embeddings_file = source_dir / "multi.wiki.bpe.vs40.d4.w2v.bin"
        write_word2vec_binary(embeddings_file, words, cls.vectors, newline_after_vector=False)

        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            tar.add(embeddings_file, arcname=f"data/multi/{embeddings_file.name}")
        cls.archive_bytes = archive.getvalue()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir_obj.cleanup()

    def setUp(self) -> None:
        self.cache_dir = Path(self.temp_dir_obj.name) / f"cache_{self._testMethodName}"
        self.model_url = "https://bpemb.h-its.org/multi/multi/multi.wiki.bpe.vs40.model"
        self.embeddings_url = "https://bpemb.h-its.org/multi/multi/multi.wiki.bpe.vs40.d4.w2v.bin.tar.gz"

    def fake_get(self, url: str, **kwargs) -> MagicMock:  # pylint: disable=unused-argument
        payloads = {self.model_url: self.model_bytes, self.embeddings_url: self.archive_bytes}
        return a_response(payloads[url])

    def create_bpemb(self, dim: int = 4) -> BPEmbBaseURLWrapperBugFix:
        with patch("deepparse.bpemb_url_bug_fix.requests.get", side_effect=self.fake_get) as get_mock:
            bpemb = BPEmbBaseURLWrapperBugFix(lang="multi", vs=self.vs, dim=dim, cache_dir=self.cache_dir)
        self.get_mock = get_mock
        return bpemb

    def test_givenAnEmptyCache_whenInit_thenDownloadTheModelAndTheEmbeddingsInTheBPEmbCacheLayout(self):
        self.create_bpemb()

        requested_urls = [call_args.args[0] for call_args in self.get_mock.call_args_list]
        self.assertEqual([self.model_url, self.embeddings_url], requested_urls)
        self.assertEqual(self.model_bytes, (self.cache_dir / "multi" / "multi.wiki.bpe.vs40.model").read_bytes())
        self.assertTrue((self.cache_dir / "multi" / "multi.wiki.bpe.vs40.d4.w2v.bin").is_file())

    def test_givenAFilledCache_whenInit_thenDoNotDownload(self):
        self.create_bpemb()

        self.create_bpemb()

        self.get_mock.assert_not_called()

    def test_whenInit_thenLoadTheVectorsAndTheDim(self):
        bpemb = self.create_bpemb()

        np.testing.assert_array_equal(self.vectors, bpemb.vectors)
        self.assertEqual(self.dim, bpemb.dim)

    def test_givenAnUnexpectedDim_whenInit_thenRaiseValueError(self):
        self.create_bpemb()
        multi_dir = self.cache_dir / "multi"
        # Cached 4-dim vectors under the 300-dim file name.
        (multi_dir / "multi.wiki.bpe.vs40.d4.w2v.bin").rename(multi_dir / "multi.wiki.bpe.vs40.d300.w2v.bin")

        with self.assertRaises(ValueError):
            BPEmbBaseURLWrapperBugFix(lang="multi", vs=self.vs, dim=300, cache_dir=self.cache_dir)

    def test_givenAnHTTPError_whenInit_thenRaiseAndDoNotCacheAPartialFile(self):
        response = a_response(b"")
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        with patch("deepparse.bpemb_url_bug_fix.requests.get", return_value=response):
            with self.assertRaises(requests.exceptions.HTTPError):
                BPEmbBaseURLWrapperBugFix(lang="multi", vs=self.vs, dim=self.dim, cache_dir=self.cache_dir)

        self.assertFalse((self.cache_dir / "multi" / "multi.wiki.bpe.vs40.model").exists())

    def test_whenPreprocess_thenLowercaseAndReplaceDigitsByZero(self):
        self.assertEqual("000 rue des lilas g0l 0b0", BPEmbBaseURLWrapperBugFix.preprocess("359 Rue des LILAS G1L 1B6"))

    def test_whenEmbed_thenReturnTheVectorsOfTheSubwordIdsOfThePreprocessedText(self):
        bpemb = self.create_bpemb()
        a_word = "Quebec2325"

        actual = bpemb.embed(a_word)

        expected_ids = self.spm.encode("quebec0000", out_type=int)
        self.assertGreater(len(expected_ids), 1)
        np.testing.assert_array_equal(self.vectors[expected_ids], actual)

    def test_whenEncode_thenReturnTheSubwordsOfThePreprocessedTextLikeBPEmb(self):
        bpemb = self.create_bpemb()

        actual = bpemb.encode("Quebec2325")

        self.assertEqual(self.spm.encode("quebec0000", out_type=str), actual)
        self.assertEqual([self.spm.piece_to_id(piece) for piece in actual], bpemb.encode_ids("Quebec2325"))

    def test_whenPickledAndUnpickled_thenEmbedTheSame(self):
        bpemb = self.create_bpemb()
        a_text = "350 rue des Lilas Ouest"

        unpickled_bpemb = pickle.loads(pickle.dumps(bpemb))

        np.testing.assert_array_equal(bpemb.embed(a_text), unpickled_bpemb.embed(a_text))
        self.assertNotIn("spm", bpemb.__getstate__())


if __name__ == "__main__":
    os.environ.setdefault("TEST_LEVEL", "unit")
    unittest.main()
