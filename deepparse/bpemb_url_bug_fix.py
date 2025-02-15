"""
Due to an error in the BPEmb base URL to download the weights, and since the authors and maintainer do not respond or
seems to maintain the project; we use a wrapper to bug-fix the URL to change it.
However, the wrapper must be placed here due to circular import.
"""

from bpemb import BPEmb


class BPEmbBaseURLWrapperBugFix(BPEmb):
    def __init__(self, **kwargs):
        self.base_url = "https://bpemb.h-its.org/multi/"
        super().__init__(**kwargs)
