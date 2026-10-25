"""
Minimal writer of the Facebook fastText ``.bin`` format (version 12, unsupervised, non-quantized) to build tiny
test models without gensim or fasttext. It follows FastText::saveModel
(https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc): the signature, Args::save,
Dictionary::save, the quant_input flag, the input DenseMatrix, the qout flag and the output DenseMatrix.
"""

import struct
from typing import Dict, List, Optional

import numpy as np

FASTTEXT_FILEFORMAT_MAGIC = 793712314
FASTTEXT_VERSION = 12

# Default fastText arguments of a cbow model with negative sampling (loss 2, model 1).
DEFAULT_ARGS = {
    "dim": 4,
    "ws": 5,
    "epoch": 5,
    "min_count": 1,
    "neg": 5,
    "word_ngrams": 1,
    "loss": 2,
    "model": 1,
    "bucket": 1000,
    "minn": 3,
    "maxn": 6,
    "lr_update_rate": 100,
    "t": 1e-4,
}

ARGS_ORDER = (
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
)

A_TINY_VOCABULARY = ["</s>", "rue", "avenue", "québec", "montréal", "123"]


def write_fasttext_bin(
    path: str,
    words: List[str],
    input_matrix: np.ndarray,
    output_matrix: Optional[np.ndarray] = None,
    args: Optional[Dict] = None,
) -> None:
    """
    Write a fastText ``.bin`` model.

    Args:
        path (str): The path of the file to write.
        words (list): The vocabulary, in the order of the input matrix rows.
        input_matrix (~numpy.ndarray): The (len(words) + bucket, dim) input matrix (words rows then buckets rows).
        output_matrix (~numpy.ndarray): The (len(words), dim) output matrix. Zeros if None.
        args (dict): The fastText arguments overriding ``DEFAULT_ARGS``.
    """
    model_args = {**DEFAULT_ARGS, **(args or {})}
    dim = model_args["dim"]
    nwords = len(words)

    input_matrix = np.ascontiguousarray(input_matrix, dtype=np.float32)
    if input_matrix.shape != (nwords + model_args["bucket"], dim):
        raise ValueError(f"The input matrix shape {input_matrix.shape} does not match the vocabulary and the args.")
    if output_matrix is None:
        output_matrix = np.zeros((nwords, dim), dtype=np.float32)
    output_matrix = np.ascontiguousarray(output_matrix, dtype=np.float32)
    # The words counts are not used to compute the vectors, so we set them to one.
    counts = [1] * nwords

    with open(path, "wb") as file:
        # FastText::signModel
        file.write(struct.pack("<2i", FASTTEXT_FILEFORMAT_MAGIC, FASTTEXT_VERSION))

        # Args::save
        file.write(struct.pack("<12i", *(model_args[name] for name in ARGS_ORDER)))
        file.write(struct.pack("<d", model_args["t"]))

        # Dictionary::save: size, nwords, nlabels, ntokens, pruneidx_size (-1 means no pruning), then the entries.
        file.write(struct.pack("<3i", nwords, nwords, 0))
        file.write(struct.pack("<2q", sum(counts), -1))
        for word, count in zip(words, counts):
            file.write(word.encode("utf-8") + b"\x00")
            # The entry type is 0 for a word (1 for a label).
            file.write(struct.pack("<qb", count, 0))

        # quant_input, then DenseMatrix::save of the input matrix (rows and cols as int64, then the floats).
        file.write(struct.pack("<?", False))
        file.write(struct.pack("<2q", *input_matrix.shape))
        file.write(input_matrix.tobytes())

        # qout, then the output matrix.
        file.write(struct.pack("<?", False))
        file.write(struct.pack("<2q", *output_matrix.shape))
        file.write(output_matrix.tobytes())


def create_tiny_fasttext_bin(path: str, words: Optional[List[str]] = None, seed: int = 42, **args) -> np.ndarray:
    """
    Write a tiny fastText model (dim 4, bucket 1000, minn 3 and maxn 6 by default) with a random input matrix.

    Return:
        The input matrix written, to compute the expected vectors by hand.
    """
    if words is None:
        words = A_TINY_VOCABULARY
    model_args = {**DEFAULT_ARGS, **args}
    random_generator = np.random.default_rng(seed)
    input_matrix = random_generator.standard_normal((len(words) + model_args["bucket"], model_args["dim"]))
    input_matrix = input_matrix.astype(np.float32)
    write_fasttext_bin(path, words, input_matrix, args=model_args)
    return input_matrix
