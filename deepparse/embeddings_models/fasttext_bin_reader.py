import mmap
import struct
from typing import Dict, List, Tuple

import numpy as np

# Constants of the Facebook fastText binary format, see FastText::checkModel and FastText::signModel in
# https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc
FASTTEXT_FILEFORMAT_MAGIC = 793712314
FASTTEXT_VERSION = 12

# Fields of Args::save (https://github.com/facebookresearch/fastText/blob/main/src/args.cc), in order:
# twelve int32 followed by one double.
ARGS_FIELDS = (
    "dim",
    "ws",
    "epoch",
    "min_count",
    "neg",
    "word_ngrams",
    "loss",
    "model",
    "bucket",
    "minn",
    "maxn",
    "lr_update_rate",
    "t",
)
ARGS_FORMAT = "<12id"

# The end of sentence token gets no subwords, see Dictionary::EOS and Dictionary::initNgrams.
EOS = b"</s>"
BOW = b"<"
EOW = b">"

_FNV_OFFSET_BASIS = 2166136261
_FNV_PRIME = 16777619
_UINT32_MASK = 0xFFFFFFFF


def fasttext_hash(token: bytes) -> int:
    """
    The fastText FNV-1a 32-bit hash (Dictionary::hash). Each byte is converted to a signed char before being
    xored, so bytes above 127 (i.e. all non-ASCII UTF-8 characters) are sign-extended to 32 bits.

    Args:
        token (bytes): The UTF-8 encoded token to hash.

    Return:
        The unsigned 32-bit hash.
    """
    h = _FNV_OFFSET_BASIS
    for byte in token:
        signed_byte = byte - 256 if byte > 127 else byte
        h = ((h ^ (signed_byte & _UINT32_MASK)) * _FNV_PRIME) & _UINT32_MASK
    return h


def compute_char_ngrams(word: bytes, minn: int, maxn: int) -> List[bytes]:
    """
    Extract the character n-grams of a word, like Dictionary::computeSubwords. The n-grams length is counted in
    UTF-8 characters (not bytes), and the single character n-grams ``<`` and ``>`` are skipped.

    Args:
        word (bytes): The UTF-8 encoded word, already wrapped with the ``<`` and ``>`` markers.
        minn (int): The minimum length of the n-grams.
        maxn (int): The maximum length of the n-grams.

    Return:
        The list of n-grams in the fastText order.
    """
    ngrams = []
    word_length = len(word)
    for i in range(word_length):
        # We skip the UTF-8 continuation bytes since a n-gram must start at a character boundary.
        if (word[i] & 0xC0) == 0x80:
            continue
        j = i
        n = 1
        while j < word_length and n <= maxn:
            j += 1
            while j < word_length and (word[j] & 0xC0) == 0x80:
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == word_length)):
                ngrams.append(word[i:j])
            n += 1
    return ngrams


def _read_struct(buffer: mmap.mmap, offset: int, fmt: str) -> Tuple[tuple, int]:
    values = struct.unpack_from(fmt, buffer, offset)
    return values, offset + struct.calcsize(fmt)


class FastTextBinReader:
    """
    Reader of the Facebook fastText ``.bin`` format (version 12, non-quantized, unsupervised models) that computes
    word vectors like ``FastText::getWordVector`` without the ``fasttext`` or ``gensim`` packages.

    The input matrix is memory-mapped, so loading is fast, the RAM usage stays low, and the pages are shared between
    processes (e.g. ``DataLoader`` workers). When pickled, only the path and the parsed dictionary are kept, and the
    memory map is reopened on unpickling.

    Args:
        path (str): Path to the fastText ``.bin`` file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._parse()
        self._open_input_matrix()

    def _parse(self) -> None:
        with open(self.path, "rb") as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
                (magic, version), offset = _read_struct(buffer, 0, "<2i")
                if magic != FASTTEXT_FILEFORMAT_MAGIC:
                    raise ValueError(f"{self.path} is not a fastText .bin file (or uses the old pre-0.9 format).")
                if version > FASTTEXT_VERSION:
                    raise ValueError(f"Unsupported fastText .bin version {version} (expected {FASTTEXT_VERSION}).")

                args_values, offset = _read_struct(buffer, offset, ARGS_FORMAT)
                self.args = dict(zip(ARGS_FIELDS, args_values))

                offset = self._parse_dictionary(buffer, offset)

                (quant_input,), offset = _read_struct(buffer, offset, "<?")
                if quant_input:
                    raise NotImplementedError("Quantized fastText models (.ftz) are not supported.")

                (rows, cols), offset = _read_struct(buffer, offset, "<2q")
                if rows != self.nwords + self.bucket or cols != self.dim:
                    raise ValueError(
                        f"Unexpected input matrix shape ({rows}, {cols}), "
                        f"expected ({self.nwords + self.bucket}, {self.dim})."
                    )
                self._matrix_shape = (rows, cols)
                self._matrix_offset = offset

    def _parse_dictionary(self, buffer: mmap.mmap, offset: int) -> int:
        # Dictionary::save: size, nwords and nlabels (int32), ntokens and pruneidx_size (int64), then for each
        # entry the word bytes, a null byte, its count (int64) and its type (int8).
        (size, nwords, nlabels), offset = _read_struct(buffer, offset, "<3i")
        if nlabels > 0:
            raise NotImplementedError("Supervised fastText models are not supported.")
        (self.ntokens, pruneidx_size), offset = _read_struct(buffer, offset, "<2q")

        self.nwords = nwords
        self._word2id: Dict[bytes, int] = {}
        for word_id in range(size):
            end = buffer.find(b"\x00", offset)
            if end == -1:
                raise ValueError(f"Truncated fastText dictionary in {self.path}.")
            self._word2id[buffer[offset:end]] = word_id
            # We skip the null byte, the count (int64) and the entry type (int8).
            offset = end + 1 + 8 + 1

        self._pruneidx: Dict[int, int] = {}
        for _ in range(max(pruneidx_size, 0)):
            (bucket_id, pruned_id), offset = _read_struct(buffer, offset, "<2i")
            self._pruneidx[bucket_id] = pruned_id
        self._pruneidx_size = pruneidx_size
        return offset

    def _open_input_matrix(self) -> None:
        self._input_matrix = np.memmap(
            self.path, dtype=np.float32, mode="r", offset=self._matrix_offset, shape=self._matrix_shape
        )

    @property
    def dim(self) -> int:
        return self.args["dim"]

    @property
    def bucket(self) -> int:
        return self.args["bucket"]

    @property
    def minn(self) -> int:
        return self.args["minn"]

    @property
    def maxn(self) -> int:
        return self.args["maxn"]

    def get_dimension(self) -> int:
        """
        Return the dimension of the word vectors (same interface as the ``fasttext`` model object).
        """
        return self.dim

    def get_word_id(self, word: str) -> int:
        """
        Return the id of a word in the dictionary, or -1 if the word is out of vocabulary.
        """
        return self._word2id.get(word.encode("utf-8"), -1)

    def get_subword_ids(self, word: str) -> List[int]:
        """
        Return the input matrix rows used to compute a word vector, like ``Dictionary::getSubwords``: the word id
        (if the word is in the vocabulary) followed by the bucket rows of its hashed character n-grams.
        """
        encoded_word = word.encode("utf-8")
        subword_ids = []
        word_id = self._word2id.get(encoded_word, -1)
        if word_id >= 0:
            subword_ids.append(word_id)
        if encoded_word != EOS:
            for ngram in compute_char_ngrams(BOW + encoded_word + EOW, self.minn, self.maxn):
                self._push_hash(subword_ids, fasttext_hash(ngram) % self.bucket)
        return subword_ids

    def _push_hash(self, subword_ids: List[int], bucket_id: int) -> None:
        # Dictionary::pushHash: a pruneidx_size of 0 means that all the n-grams were pruned, and -1 no pruning.
        if self._pruneidx_size == 0:
            return
        if self._pruneidx_size > 0:
            if bucket_id not in self._pruneidx:
                return
            bucket_id = self._pruneidx[bucket_id]
        subword_ids.append(self.nwords + bucket_id)

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Compute the vector of a word as the mean of its subword rows, or a zero vector if it has none.
        """
        subword_ids = self.get_subword_ids(word)
        if not subword_ids:
            return np.zeros(self.dim, dtype=np.float32)
        return self._input_matrix[subword_ids].mean(axis=0, dtype=np.float32)

    def __getitem__(self, word: str) -> np.ndarray:
        return self.get_word_vector(word)

    def __getstate__(self) -> dict:
        # We do not pickle the memory map; it is reopened from the path in __setstate__.
        state = self.__dict__.copy()
        del state["_input_matrix"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_input_matrix()
