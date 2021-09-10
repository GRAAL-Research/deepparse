from dataclasses import dataclass
from typing import Dict

from .formatted_compared_addresses import FormattedComparedAddresses


@dataclass
class FormattedComparedAddressesTags(FormattedComparedAddresses):
    """class that inherits from abstract class FormattedComparedAddresses and implements its comparison report."""

    def get_probs(self) -> Dict:
        """get probs of tags for the parsing made with deepparse

        Returns:
            Dict: the key is the raw address and the value is the tags with their associated probabilities.
        """
        return {self.address_two.raw_address: self.address_two.address_parsed_components}

    def _comparison_report_builder(self) -> str:
        """Builds a formatted string that represents a comparison report for addresses tags comparison

        Returns:
            str: A formatted string that represents a comparison report for addresses tags comparison
        """

        formatted_str = ""
        intro_str = "Comparison report of tags for parsed address: "
        if self.identical:
            formatted_str +=  intro_str + "Identical\n"
        else:
            formatted_str +=  intro_str + "Not identical\n"
        formatted_str += "Raw address: " + self.address_one.raw_address +"\n\n"

        formatted_str += "Tags: \n"
        formatted_str += self.origin[0] + ": " + str(self.address_one.to_list_of_tuples()) + "\n\n"
        formatted_str += self.origin[1] + ": " + str(self.address_two.to_list_of_tuples()) + "\n\n"

        formatted_str += self._get_probs_of_tags()

        if not self.identical:
            formatted_str += "\n\n"
            formatted_str += "Addresses tags differences between the two parsing:\n"
            formatted_str += self._get_tags_diff_color(self.origin[0], self.origin[1])
        
        return formatted_str
