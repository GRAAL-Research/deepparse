import re
from typing import List

# The first group is the unit, and the second is the street number.
# Both include letters since they can include letters in some countries. For example,
# unit 3a or address 305a.
hyphen_splitted_unit_and_street_number_regex = r"^([0-9]*[a-z]?)-([0-9]*[a-z]?) "


class AddressCleaner:
    def __init__(self, with_hyphen_split: bool = False) -> None:
        self.with_hyphen_split = with_hyphen_split

    def clean(self, addresses: List[str]) -> List[str]:
        res = []

        for address in addresses:
            processed_address = self.coma_cleaning(address)

            processed_address = self.lower_cleaning(processed_address)

            if self.with_hyphen_split:
                processed_address = self.hyphen_cleaning(processed_address)

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
        # See issue 137 for more details https://github.com/GRAAL-Research/deepparse/issues/137.
        # Since some addresses use the hyphen to split the unit and street address, we replace the hyphen
        # with whitespaces to allow a proper splitting of the address.
        # For example, the proper parsing of the address 3-305 street name is
        # Unit: 3, StreetNumber: 305, StreetName: street name.
        # Note: the hyphen is also used in some cities' names, such as Saint-Jean; thus, we use regex to detect
        # the proper hyphen to replace.
        return re.sub(hyphen_splitted_unit_and_street_number_regex, r"\1 \2 ", text)
