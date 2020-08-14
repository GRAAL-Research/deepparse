from typing import Dict


class TargetConverter:
    """
    Class to define logic of tag to idx conversion

    Args:
        tags_to_idx (Dict): A dictionary where the value are the tags (e.g. StreetNumber) and the value are
            the idx (int) (e.g. 1).
    """

    def __init__(self, tags_to_idx: Dict) -> None:
        self.tags_to_idx = tags_to_idx

    def __call__(self, target_tag: str) -> int:
        return self.tags_to_idx[target_tag]
