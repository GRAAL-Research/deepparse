from typing import List


class AddressCleaner:
    def clean(self, addresses: List[str]) -> List[str]:
        res = []

        for address in addresses:
            processed_address = self.coma_cleaning(address)

            processed_address = self.hyphen_cleaning(processed_address)

            processed_address = self.lower_cleaning(processed_address)

            res.append(" ".join(processed_address.split()))
        return res

    @staticmethod
    def coma_cleaning(text: str) -> str:
        # See issue 56 https://github.com/GRAAL-Research/deepparse/issues/56
        return text.replace(",", "")

    @staticmethod
    def lower_cleaning(text: str) -> str:
        # Since the original training data was in lowercase
        return text.lower()

    @staticmethod
    def hyphen_cleaning(text: str) -> str:
        # Since some addresses use the hyphen to split the unit and street address
        # For example, the proper parsing of the address 3-305 street name is
        # Unit: 3, StreetNumber: 305, StreetName: street name.
        # We replace the hyphen with whitespaces since we use it for splitting,
        # thus, it will allow a proper splitting of the address.
        # See issue 137 for more details https://github.com/GRAAL-Research/deepparse/issues/137.
        return text.replace("-", " ")
