from dataclasses import dataclass
from typing import Dict

from .formatted_compared_addresses import FormattedComparedAddresses


@dataclass(frozen=True)
class FormattedComparedAddressesTags(FormattedComparedAddresses):
    """
    A formatted compared address of two already tagged addresses.
    """

    def _get_probs(self) -> Dict:
        """
        Get the tags from the parsing with their associated probabilities, the method needs to be implemented in each
        class because they don't use the probabilities the same way.
        """
        return {
            self.origin[0]: self.first_address.address_parsed_components,
            self.origin[1]: self.second_address.address_parsed_components
        }

    def _get_probs_of_tags(self, verbose: bool = True) -> str:
        """
        Takes the tags and their probabilities for the report.
        """
        formatted_str = ""
        if verbose:
            formatted_str += "Probabilities of parsed tags: \n"
        for index, tuple_dict in enumerate(self._get_probs().items()):
            key, value = tuple_dict
            formatted_str += key + ": "
            formatted_str += str(value) + "\n"
            if index == 0:
                formatted_str += "\n"
        return formatted_str

    def _comparison_report_builder(self) -> str:
        """
        Builds the core of a comparison report for the different comparisons. Since the procedure to make a tags
        comparison and the raw addresses comparison is different, the comparison report is not the same for the two.
        It is then implemented in each specific class.
        """

        formatted_str = ""
        intro_str = "Comparison report of tags for parsed address: "
        if self.identical:
            formatted_str += intro_str + "Identical\n\n"
        else:
            formatted_str += intro_str + "Not equivalent\n\n"
        formatted_str += "Raw address: " + self.first_address.raw_address + "\n\n\n"

        formatted_str += "Tags: \n"
        formatted_str += self.origin[0] + ": " + str(self.first_address.to_list_of_tuples()) + "\n\n"
        formatted_str += self.origin[1] + ": " + str(self.second_address.to_list_of_tuples()) + "\n\n\n"

        if self.with_prob:
            formatted_str += self._get_probs_of_tags()

        if not self.identical:
            formatted_str += "\n"
            formatted_str += "Addresses tags differences between the two parsing:\n"
            formatted_str += self._get_tags_diff_color(self.origin[0], self.origin[1])

        return formatted_str
