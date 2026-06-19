import contextlib
import logging
import ssl
import warnings
from pathlib import Path
from typing import Any, Dict

import requests
from numpy import ndarray
from urllib3.exceptions import InsecureRequestWarning

from ..bpemb_url_bug_fix import BPEmbBaseURLWrapperBugFix
from .embeddings_model import EmbeddingsModel

logger = logging.getLogger(__name__)


class BPEmbEmbeddingsModel(EmbeddingsModel):
    """
    BPEmb embeddings network from `BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages
    <https://www.aclweb.org/anthology/L18-1473/>`_. The arguments are the same as the
    `BPEmb class <https://github.com/bheinzerling/bpemb/blob/master/bpemb/bpemb.py>`_

    Params:
        cache_dir (str): Path to the cache directory to the embeddings' bin vector and the model.
        verbose (bool): Whether or not to make the loading of the embeddings verbose.
    """

    def __init__(self, cache_dir: str, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)
        with warnings.catch_warnings():
            # annoying scipy.sparcetools private module warnings removal
            # annoying boto warnings
            warnings.filterwarnings("ignore")
            self.model = self._load_bpemb_model(cache_dir)

    def _load_bpemb_model(self, cache_dir: str) -> BPEmbBaseURLWrapperBugFix:
        """
        Download (if needed) and load the BPEmb model. We download with SSL verification enabled. The BPEmb
        host (set by :class:`BPEmbBaseURLWrapperBugFix`) currently has a valid certificate, but it has broken
        before (`bpemb issue 63 <https://github.com/bheinzerling/bpemb/issues/63>`_). As a failsafe, and only
        if an SSL error occurs, we retry once with verification disabled while warning loudly, so a transient
        upstream certificate problem does not break Deepparse outright.

        We use the default parameters other than the dim at 300 and a vs of 100,000.
        """
        model_kwargs = {"lang": "multi", "vs": 100000, "dim": 300, "cache_dir": Path(cache_dir)}
        try:
            return BPEmbBaseURLWrapperBugFix(**model_kwargs)
        except (requests.exceptions.SSLError, ssl.SSLError) as ssl_error:
            logger.warning(
                "SSL verification failed while downloading the BPEmb embeddings. Retrying once without SSL "
                "verification as a failsafe; this is insecure (man-in-the-middle risk). If it persists, the "
                "BPEmb host certificate is likely broken again (see bpemb issue 63). Original error: %s",
                ssl_error,
            )
            with no_ssl_verification():
                return BPEmbBaseURLWrapperBugFix(**model_kwargs)

    def __call__(self, word: str) -> ndarray:
        """
        Callable method to get a word vector.

        Args:
            word (str): Word to get vector.

        Return:
            The BP embedding for a word.
        """
        return self.model.embed(word)


@contextlib.contextmanager
def no_ssl_verification():
    """Context Manager to disable SSL verification when using ``requests`` library.

    Reference: https://gist.github.com/ChenTanyi/0c47652bd916b61dc196968bca7dad1d.

    It will be removed when https://github.com/bheinzerling/bpemb/issues/63 is resolved.
    """
    opened_adapters = set()
    old_merge_environment_settings = requests.Session.merge_environment_settings

    def merge_environment_settings(  # pylint: disable=R0913
        self, url: str, proxies: Dict[str, Any], stream: bool, verify: bool, cert: Any
    ) -> Dict[str, Any]:
        # Verification happens only once per connection, so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings["verify"] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except Exception:  # pylint: disable=broad-except
                pass
