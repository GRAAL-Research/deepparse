from typing import Dict, Union


class TagsConverter:
    """
    Class to define the logic of tag to idx conversion and vice versa.

    Args:
        tags_to_idx (Dict): A dictionary where the keys are the tags (e.g. StreetNumber) and the values are
            the indexes (int) (e.g. 1).
    """

    def __init__(self, tags_to_idx: Dict) -> None:
        self.tags_to_idx = tags_to_idx
        self.idx_to_tags = {v: k for k, v in tags_to_idx.items()}

    def __call__(self, key: Union[str, int]) -> int:
        """
        If it is a ``str, ``, it will convert from a "tag" to an IDX, and if ``int``, it will convert from an IDX to
        a "tag" using the convert table.
        """
        if isinstance(key, str):
            return self.tags_to_idx[key]
        return self.idx_to_tags[key]

    @property
    def dim(self) -> int:
        return len(self.tags_to_idx)
