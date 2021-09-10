from dataclasses import dataclass
from typing import Dict

from .formatted_compared_addresses import FormattedComparedAddresses


@dataclass
class FormattedComparedAddressesRaw(FormattedComparedAddresses):

    def get_probs(self) -> Dict:
        """get probs of tags for the parsing made with deepparse

        Returns:
            Dict: the key is the raw address and the value is the tags with thei associated probabilities.
        """
        return {self.address_one.raw_address: self.address_one.address_parsed_components,
                self.address_two.raw_address: self.address_two.address_parsed_components}

    def _get_raw_diff_color(self, verbose=True) -> str:
        """Print the raw addresses and highlight the differences between them."""

        str_formattted = ""

        if verbose:
            str_formattted +="White: Shared\n"
            if not self.colorblind:
                str_formattted += "Red: Belongs only to address one\n"
                str_formattted += "Green: Belongs only to address two\n"
            else:
                str_formattted += "Blue: Belongs only to address one\n"
                str_formattted += "Yellow: Belongs only to address two\n"
            str_formattted += "\n"
        str_formattted += self._get_color_diff(self.address_one.raw_address, self.address_two.raw_address, highlight=True) + "\n"
        return str_formattted

    def _comparison_report_builder(self) -> str:
        """Builds a formatted string that represents a comparison report for raw addresses comparison

        Returns:
            str: A formatted string that represents a comparison report for raw addresses comparison
        """
        str_formattted = ""
        intro_str = "Comparison report of the two raw addresses: "
        if self.identical:
            str_formattted += intro_str + "Identical\n"
        else:
            if self.equivalent:
                str_formattted += intro_str + "Equivalent\n"
            else:
                str_formattted += intro_str + "Not equivalent\n"
        str_formattted += "\n"
        str_formattted += "Address one: " + self.address_one.raw_address + "\n"
        str_formattted += "and\n"
        str_formattted += "Address two: " + self.address_two.raw_address + "\n\n\n"
        str_formattted += "Probabilities of parsed tags for the addresses with " + self.origin[0] + ": \n\n"
        probs = list(self.get_probs().values())
        str_formattted += "Parsed address: " + repr(self.address_one) + "\n"
        str_formattted += str(probs[0]) + "\n"
        if not self.identical:
            
            str_formattted += "\nParsed address: " + repr(self.address_two) +"\n"
            str_formattted += str(probs[1]) + "\n"

            if self.equivalent:

                str_formattted += "\n\nRaw differences between the two addresses: \n"
                str_formattted += self._get_raw_diff_color()
            else:

                str_formattted += "\n\nAddresses tags differences between the two addresses: \n"
                str_formattted += self._get_tags_diff_color()

        return str_formattted
