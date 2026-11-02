"""
Minimal reimplementation of the parts of `BPEmb <https://github.com/bheinzerling/bpemb>`_ that Deepparse uses: the
download of the multilingual model and ``embed``. The ``bpemb`` package is not maintained (we already had to fix its
base URL) and depends on ``gensim``, which does not support Python 3.14. We keep the cache layout
(``<cache_dir>/multi/...``) of the former wrapper, so existing caches are reused.

It must be placed here (and not in the embeddings_models package) due to circular import: ``download_tools`` uses
it, and the ``embeddings_models`` package imports ``download_tools`` through ``deepparse``.
"""

import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import requests
import sentencepiece

__all__ = ["BPEmbLoader", "load_word2vec_binary_vectors"]


def load_word2vec_binary_vectors(path: Path) -> np.ndarray:
    """
    Load the vectors of a word2vec binary file (``"<count> <dim>\\n"`` header, then for each word its bytes, a
    space and ``dim`` little-endian float32), in file order. The words are not kept since BPEmb indexes the vectors by
    their SentencePiece id, which is their position in the file.
    """
    content = Path(path).read_bytes()
    header_end = content.index(b"\n")
    count, dim = (int(value) for value in content[:header_end].split())
    vector_size = dim * np.dtype(np.float32).itemsize

    vectors = np.empty((count, dim), dtype=np.float32)
    position = header_end + 1
    for index in range(count):
        # Some writers add a newline after each vector, which is not part of the next word.
        while content[position : position + 1] == b"\n":
            position += 1
        position = content.index(b" ", position) + 1
        vectors[index] = np.frombuffer(content, dtype="<f4", count=dim, offset=position)
        position += vector_size
    return vectors


class BPEmbLoader:
    """
    Gensim-free loader of the multilingual BPEmb subword embeddings (SentencePiece segmentation and word2vec
    vectors). It downloads the model and the embeddings on the first use and caches them.

    Args:
        lang (str): The BPEmb language. Deepparse only uses ``"multi"``.
        vs (int): The vocabulary size.
        dim (int): The embeddings dimension.
        cache_dir (Path): The directory where the model and the embeddings are (or will be) downloaded. By default,
            the BPEmb cache directory (``~/.cache/bpemb``).

    Note:
        It has the BPEmb attributes and methods Deepparse and most users rely on (``embed``, ``encode``,
        ``encode_ids``, ``vectors``, ``dim``, ``vs``, ``lang`` and ``spm``), but not the gensim ``emb`` attribute.
    """

    base_url = "https://bpemb.h-its.org/multi/"
    model_template = "{lang}/{lang}.wiki.bpe.vs{vs}.model"
    embeddings_template = "{lang}/{lang}.wiki.bpe.vs{vs}.d{dim}.w2v.bin"
    archive_suffix = ".tar.gz"

    def __init__(
        self, *, lang: str, vs: int = 10000, dim: int = 100, cache_dir: Path = Path.home() / ".cache" / "bpemb"
    ) -> None:
        self.lang = lang
        self.vs = vs
        self.cache_dir = Path(cache_dir)

        self.model_file = self._download(self.model_template.format(lang=lang, vs=vs))
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(self.model_file))

        self.embeddings_file = self._download(self.embeddings_template.format(lang=lang, vs=vs, dim=dim), archive=True)
        self.vectors = load_word2vec_binary_vectors(self.embeddings_file)
        self.dim = self.vectors.shape[1]
        if self.dim != dim:
            raise ValueError(f"The BPEmb embeddings dimension is {self.dim}, but {dim} was expected.")

    def _download(self, file: str, archive: bool = False) -> Path:
        cached_file = self.cache_dir / file
        if cached_file.exists():
            return cached_file

        cached_file.parent.mkdir(parents=True, exist_ok=True)
        url = self.base_url + file + (self.archive_suffix if archive else "")
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded_file = Path(temp_dir) / "download"
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(downloaded_file, "wb") as file_handler:
                    shutil.copyfileobj(response.raw, file_handler)

            if archive:
                # The archive contains a single file; we ignore its directory, like BPEmb.
                with tarfile.open(downloaded_file) as tar:
                    member = next(member for member in tar.getmembers() if member.isfile())
                    with tar.extractfile(member) as extracted_file, open(cached_file, "wb") as file_handler:
                        shutil.copyfileobj(extracted_file, file_handler)
            else:
                shutil.move(downloaded_file, cached_file)
        return cached_file

    @staticmethod
    def preprocess(text: str) -> str:
        """
        BPEmb preprocessing: the pretrained BPE models were trained on lowercased text with the digits replaced by 0.
        """
        return re.sub(r"\d", "0", text.lower())

    def encode(self, text: str) -> List[str]:
        """
        Byte-pair encode a text into its subwords.
        """
        return self.spm.encode(self.preprocess(text), out_type=str)

    def encode_ids(self, text: str) -> List[int]:
        """
        Byte-pair encode a text into its subword ids.
        """
        return self.spm.encode(self.preprocess(text), out_type=int)

    def embed(self, text: str) -> np.ndarray:
        """
        Byte-pair encode a text and return its subword embeddings, a matrix of shape (number of subwords, dim).
        """
        return self.vectors[self.encode_ids(text)]

    def __getstate__(self) -> dict:
        # The SentencePiece processor is reloaded from its file when unpickled (e.g. in a spawned worker).
        state = self.__dict__.copy()
        del state["spm"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(self.model_file))
