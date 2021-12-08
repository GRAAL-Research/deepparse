# pylint: disable=line-too-long
from typing import List


class AddressCleaner:

    @staticmethod
    def clean(addresses: List[str]):
        res = []

        for address in addresses:
            processed_address = address.replace(
                ",", '')  # see issue 56 https://github.com/GRAAL-Research/deepparse/issues/56

            processed_address = processed_address.lower()  # since the original training data was in lowercase

            res.append(" ".join(processed_address.split()))

        return res
