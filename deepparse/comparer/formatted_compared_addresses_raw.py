from dataclasses import dataclass
from typing import Dict

from .formatted_compared_addresses import FormattedComparedAddresses


@dataclass(frozen=True)
class FormattedComparedAddressesRaw(FormattedComparedAddresses):
    """class that inherits from abstract class FormattedComparedAddresses and implements its methods in order to build
        a comparison report."""

    def _get_probs(self) -> Dict:
        """get the tags from the parsing with their associated probabilities

        Returns:
            Dict: the key is the raw address and the value is the tags with thei associated probabilities.
        """
        return {self.first_address.raw_address: self.first_address.address_parsed_components,
                self.second_address.raw_address: self.second_address.address_parsed_components}

    def _get_raw_diff_color(self, verbose=True) -> str:
        """Print the raw addresses and highlight the differences between them."""

        str_formattted = ""

        if verbose:
            str_formattted +="White: Shared\n"
            str_formattted += "Blue: Belongs only to the first address\n"
            str_formattted += "Yellow: Belongs only to the second address\n"
            str_formattted += "\n"

        str_formattted += self._get_color_diff(self.first_address.raw_address, self.second_address.raw_address, highlight=True) + "\n"
        return str_formattted

    def _comparison_report_builder(self) -> str:
        """Builds a formatted string that represents a comparison report for raw addresses comparison

        Returns:
            str: A formatted string that represents a comparison report for raw addresses comparison
        """
        str_formattted = ""
        intro_str = "Comparison report of the two raw addresses: "
        if self.identical:
            str_formattted += intro_str + "Identical\n\n"
            str_formattted += "Address : " + self.first_address.raw_address + "\n\n\n"
        else:
            if self.equivalent:
                str_formattted += intro_str + "Equivalent\n\n"
            else:
                str_formattted += intro_str + "Not equivalent\n\n"
        
        
            str_formattted += "First address : " + self.first_address.raw_address + "\n"
            str_formattted += "and\n"
            str_formattted += "Second address: " + self.second_address.raw_address + "\n\n\n"
        str_formattted += "Probabilities of parsed tags for the addresses with " + self.origin[0] + ": \n"
        probs = list(self._get_probs().values())
        str_formattted += "Parsed address: " + repr(self.first_address) + "\n"
        str_formattted += str(probs[0]) + "\n"
        if not self.identical:
            
            str_formattted += "\nParsed address: " + repr(self.second_address) +"\n"
            str_formattted += str(probs[1]) + "\n"

            if self.equivalent:

                str_formattted += "\n\nRaw differences between the two addresses: \n"
                str_formattted += self._get_raw_diff_color()
            else:

                str_formattted += "\n\nAddresses tags differences between the two addresses: \n"
                str_formattted += self._get_tags_diff_color()

        return str_formattted
