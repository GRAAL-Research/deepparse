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

    def comparison_report(self) -> None:
        """print a comparison report for addresses tags comparison"""
        # get terminal size to adapt the output to the user
        # nb_delimiters = os.get_terminal_size().columns if nb_delimiters is None else nb_delimiters
        nb_delimiters = 125

        comparison_report_signal = "=" * nb_delimiters
        print(comparison_report_signal)

        intro_str = "Comparison report of tags for parsed address: "
        if self.identical:
            print(intro_str + "Identical")
        else:
            print(intro_str + "Not identical")
        print("Raw address: " + self.address_one.raw_address)

        print("")
        print("Tags: ")
        print(self.origin[0] + ": ", self.address_one.to_list_of_tuples())
        print("")
        print(self.origin[1] + ": ", self.address_two.to_list_of_tuples())
        print("")
        print("")

        self._print_probs_of_tags()

        if not self.identical:
            print("")
            print("")
            print("Addresses tags differences between the two parsing:")
            self._print_tags_diff_color()
        print("")

        print(comparison_report_signal)
        print("")
