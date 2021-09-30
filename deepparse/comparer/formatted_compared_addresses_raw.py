from dataclasses import dataclass
from typing import Dict

from .formatted_compared_addresses import FormattedComparedAddresses


@dataclass(frozen=True)
class FormattedComparedAddressesRaw(FormattedComparedAddresses):
    """
    A formatted compared address of two raw (not parsed) addresses.
    """

    def _get_probs(self) -> Dict:
        """
        Get the tags from the parsing with their associated probabilities, the method needs to be implemented in each
        class because they don't use the probabilities the same way.
        """
        return {
            self.first_address.raw_address: self.first_address.address_parsed_components,
            self.second_address.raw_address: self.second_address.address_parsed_components
        }

    def _get_raw_diff_color(self, verbose=True) -> str:
        """
        Print the raw addresses and highlight the differences between them.
        """

        str_formatted = ""

        if verbose:
            str_formatted += "White: Shared\n"
            str_formatted += "Blue: Belongs only to the first address\n"
            str_formatted += "Yellow: Belongs only to the second address\n"
            str_formatted += "\n"

        str_formatted += self._get_color_diff(
            self.first_address.raw_address, self.second_address.raw_address, highlight=True) + "\n"
        return str_formatted

    def _comparison_report_builder(self) -> str:
        """
        Builds the core of a comparison report for the different comparisons. Since the procedure to make a tags
        comparison and the raw addresses comparison is different, the comparison report is not the same for the two.
        It is then implemented in each specific class.
        """
        str_formatted = ""
        intro_str = "Comparison report of the two raw addresses: "
        if self.identical:
            str_formatted += intro_str + "Identical\n\n"
            str_formatted += "Address : " + self.first_address.raw_address + "\n\n\n"
        else:
            if self.equivalent:
                str_formatted += intro_str + "Equivalent\n\n"
            else:
                str_formatted += intro_str + "Not equivalent\n\n"

            str_formatted += "First address : " + self.first_address.raw_address + "\n"
            str_formatted += "and\n"
            str_formatted += "Second address: " + self.second_address.raw_address + "\n\n\n"
        str_formatted += "Probabilities of parsed tags for the addresses with " + self.origin[0] + ": \n\n"
        probs = list(self._get_probs().values())
        str_formatted += "Parsed address: " + repr(self.first_address) + "\n"
        str_formatted += str(probs[0]) + "\n"
        if not self.identical:

            str_formatted += "\nParsed address: " + repr(self.second_address) + "\n"
            str_formatted += str(probs[1]) + "\n"

            if self.equivalent:

                str_formatted += "\n\nRaw differences between the two addresses: \n"
                str_formatted += self._get_raw_diff_color()
            else:

                str_formatted += "\n\nAddresses tags differences between the two addresses: \n"
                str_formatted += self._get_tags_diff_color()

        return str_formatted
