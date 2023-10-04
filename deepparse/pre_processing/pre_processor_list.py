from typing import List


class PreProcessorList:
    """
    A list of pre-processor address cleaner that apply them in batch over a list of addresses.
    """

    def __init__(self, pre_processors: List):
        """

        Args:
            pre_processors: (list) A list of pre-processor address cleaner.
        """
        self.pre_processors = pre_processors

    def apply(self, addresses: List[str]) -> List[str]:
        """
        Apply the pre-processors address cleaner over a list of address.
        Args:
            addresses: (list) a list of address.

        Returns: a list of cleaned address.
        """
        res = []

        for address in addresses:
            processed_address = address

            for pre_processor in self.pre_processors:
                processed_address = pre_processor(processed_address)

            res.append(" ".join(processed_address.split()))
        return res
