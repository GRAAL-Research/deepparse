import contextlib
import warnings
from pathlib import Path

import requests
from bpemb import BPEmb

from numpy.core.multiarray import ndarray
from urllib3.exceptions import InsecureRequestWarning

from .embeddings_model import EmbeddingsModel


class BPEmbEmbeddingsModel(EmbeddingsModel):
    """
    BPEmb embeddings network from `BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages
    <https://www.aclweb.org/anthology/L18-1473/>`_. The arguments are the same as the
    `BPEmb class <https://github.com/bheinzerling/bpemb/blob/master/bpemb/bpemb.py>`_

    Params:
        cache_dir (str): Path to the cache directory to the embeddings' bin vector and the model.
        verbose (bool): Wether or not to make the loading of the embeddings verbose.
    """

    def __init__(self, cache_dir: str, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)
        with warnings.catch_warnings():
            # annoying scipy.sparcetools private module warnings removal
            # annoying boto warnings
            warnings.filterwarnings("ignore")
            # hotfix until https://github.com/bheinzerling/bpemb/issues/63
            # is resolved.
            with no_ssl_verification():
                model = BPEmb(lang="multi", vs=100000, dim=300, cache_dir=Path(cache_dir))  # defaults parameters
        self.model = model

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

    Will be removed when https://github.com/bheinzerling/bpemb/issues/63 is resolved.
    """
    opened_adapters = set()
    old_merge_environment_settings = requests.Session.merge_environment_settings

    def merge_environment_settings(self, url, proxies, stream, verify, cert):  # pylint: disable=R0913
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
            except BaseException:  # pylint: disable=W0702, W0703
                pass
