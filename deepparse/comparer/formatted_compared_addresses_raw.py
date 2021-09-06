from dataclasses import dataclass
from .formatted_compared_addresses import FormatedComparedAddresses
import sys

@dataclass
class FormattedComparedAddressesRaw(FormatedComparedAddresses):

    def get_probs(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        return {self.address_one.raw_address: self.address_one.address_parsed_components,
                self.address_two.raw_address: self.address_two.address_parsed_components}


    def _print_raw_diff_color(self, verbose = True) -> None:
        """Print the raw addresses and highlight the differences between them.

        Raises:
            ValueError: Can only use the method if the comparison is for raw addresses, meaning at least
            two raw addresses has been passed in arguments.
        """


        result = self._get_color_diff(self.address_one.raw_address, self.address_two.raw_address, highlight=True)

        if verbose:
            print("White: Shared")
            if not self.metadata["colorblind"]:
                print("Red: Belongs only to address one")
                print("Green: Belongs only to address two")
            else:
                print("Blue: Belongs only to address one")
                print("Yellow: Belongs only to address two")
            print("")
        sys.stdout.writelines(result)
        print("")


    def comparison_report(self):
        """[summary]

        Raises:
            ValueError: [description]
        """
        # get terminal size to adapt the output to the user
        # nb_delimiters = os.get_terminal_size().columns if nb_delimiters is None else nb_delimiters
        nb_delimiters = 125

        comparison_report_signal = "=" * nb_delimiters
        print(comparison_report_signal)

        intro_str = "Comparison report of the two raw addresses: "
        if self.indentical:
            print(intro_str +  "Identical")
        else:
            if self.equivalent:
                print(intro_str +  "Equivalent")
            else:
                print(intro_str +  "Not equivalent")
        print("")
        print("Address one: " + self.address_one.raw_address)
        print("and")
        print("Address two: " +self.address_two.raw_address)
        print("")



        print("")
        print("Probabilities of parsed tags for the addresses with "+self.metadata["origin"][0] +": ")
        print("")
        probs = list(self.get_probs().values())
        print("Parsed address: "+ self.metadata["repr"][0])
        print(probs[0])
        if not self.indentical:
            print("")
            print("Parsed address: "+ self.metadata["repr"][1])
            print(probs[1])

            if self.equivalent:
                print("")
                print("")
                print("Raw differences between the two addresses: ")
                self._print_raw_diff_color()
            else:
                print("")
                print("")
                print("Addresses tags differences between the two addresses: ")
                self._print_tags_diff_color()
        print(comparison_report_signal)