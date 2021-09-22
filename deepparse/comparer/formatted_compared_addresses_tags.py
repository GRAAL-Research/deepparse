from dataclasses import dataclass
from typing import Dict

from .formatted_compared_addresses import FormattedComparedAddresses


@dataclass(frozen=True)
class FormattedComparedAddressesTags(FormattedComparedAddresses):
    """class that inherits from abstract class FormattedComparedAddresses and implements its comparison report."""

    
    def _get_probs(self) -> Dict:
        """get the tags from the parsing with their associated probabilities

        Returns:
            Dict: the key is the raw address and the value is the tags with thei associated probabilities.
        """
        return {self.origin[0]: self.first_address.address_parsed_components,
                self.origin[1]: self.second_address.address_parsed_components}

    def _get_probs_of_tags(self, verbose:bool = True) -> str:
        """takes the tags and their probabilities and print them to console
        Args:
            verbose (bool, optional): If true, the results are presented. Defaults to True.
        """
        formatted_str = ""
        if verbose:
            formatted_str += "Probabilities of parsed tags: \n"
        for index, tuple_dict in enumerate(self._get_probs().items()):
            key, value = tuple_dict
            formatted_str += key + ": "
            formatted_str += str(value) + "\n\n"
            if index > 0:
                formatted_str += "\n"
        return formatted_str

    def _comparison_report_builder(self) -> str:
        """Builds a formatted string that represents a comparison report for addresses tags comparison

        Returns:
            str: A formatted string that represents a comparison report for addresses tags comparison
        """

        formatted_str = ""
        intro_str = "Comparison report of tags for parsed address: "
        if self.identical:
            formatted_str +=  intro_str + "Identical\n\n"
        else:
            formatted_str +=  intro_str + "Not identical\n\n"
        formatted_str += "Raw address: " + self.first_address.raw_address +"\n\n"

        formatted_str += "Tags: \n"
        formatted_str += self.origin[0] + ": " + str(self.first_address.to_list_of_tuples()) + "\n\n"
        formatted_str += self.origin[1] + ": " + str(self.second_address.to_list_of_tuples()) + "\n\n\n"


        if self.with_probs:
            formatted_str += self._get_probs_of_tags()

        if not self.identical:
            formatted_str += "Addresses tags differences between the two parsing:\n"
            formatted_str += self._get_tags_diff_color(self.origin[0], self.origin[1])
        
        return formatted_str
