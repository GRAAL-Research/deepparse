from typing import List


class AddressCleaner:

    def clean(self, addresses: List[str]) -> List[str]:
        res = []

        for address in addresses:
            processed_address = self.coma_cleaning(address)

            processed_address = self.lower_cleaning(processed_address)

            res.append(" ".join(processed_address.split()))
        return res

    @staticmethod
    def coma_cleaning(text: str) -> str:
        # See issue 56 https://github.com/GRAAL-Research/deepparse/issues/56
        return text.replace(",", '')

    @staticmethod
    def lower_cleaning(text: str) -> str:
        # Since the original training data was in lowercase
        return text.lower()
