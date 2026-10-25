"""
Deprecated module kept for backward compatibility. Use :mod:`deepparse.bpemb_loader` instead.

.. deprecated:: 0.12.0
    ``BPEmbBaseURLWrapperBugFix`` was renamed ``BPEmbLoader`` (in ``deepparse.bpemb_loader``). This module and its
    alias will be removed in 1.0.0.
"""

import warnings
from pathlib import Path

from .bpemb_loader import BPEmbLoader, load_word2vec_binary_vectors

__all__ = ["BPEmbBaseURLWrapperBugFix", "load_word2vec_binary_vectors"]


class BPEmbBaseURLWrapperBugFix(BPEmbLoader):
    """
    Deprecated alias of :class:`~deepparse.bpemb_loader.BPEmbLoader`, with the same arguments and behaviour.

    .. deprecated:: 0.12.0
        Use ``deepparse.BPEmbLoader`` instead. This alias will be removed in 1.0.0.
    """

    def __init__(
        self, *, lang: str, vs: int = 10000, dim: int = 100, cache_dir: Path = Path.home() / ".cache" / "bpemb"
    ) -> None:
        warnings.warn(
            "BPEmbBaseURLWrapperBugFix is deprecated and will be removed in 1.0.0, use deepparse.BPEmbLoader "
            "instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(lang=lang, vs=vs, dim=dim, cache_dir=cache_dir)
