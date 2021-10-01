import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Union, Tuple, Dict

from ..parser import FormattedParsedAddress


@dataclass(frozen=True)
class FormattedComparedAddresses(ABC):
    """
    Abstract method that defined a comparison for addresses returned by the address comparer.

    Args:
        first_address(FormattedParsedAddress): A formatted parsed address that contains the parsing information
                                                for the first one.
        second_address(FormattedParsedAddress): A formatted parsed address that contains the parsing information
                                                for the second one.
        origin: (Tuple[str, str]): The origin of the parsing (ex : from source or from a deepparse pre-trained model).

    Example:

        .. code-block:: python

            address_comparer = AddressesComparer(AddressParser())
            raw_identical_comparison = address_comparer.compare_raw(
                                                        ("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6",
                                                        "450 rue des Lilas Ouest Quebec city Quebec G1L 1B6"))
    """
    first_address: FormattedParsedAddress
    second_address: FormattedParsedAddress
    origin: Tuple[str]
    with_prob: bool

    @property
    def list_of_bool(self) -> List:
        """
        A list of boolean that contains all the address components names and indicates if it is the same for the
        two addresses.

        Return:
            A list of the boolean.
        """
        return self._bool_address_tags_are_the_same(
            [self.first_address.to_list_of_tuples(),
             self.second_address.to_list_of_tuples()])

    @property
    def equivalent(self) -> bool:
        """
        Check if the parsing is the same for the two addresses.

        Return:
            A bool.
        """
        return all((bool_address[1] for bool_address in self.list_of_bool))

    @property
    def identical(self) -> bool:
        """
        Check if the parsing is the same for the two addresses and if the raw addresses are identical.

        Return:
            A bool.
        """
        is_identical = False
        if self.equivalent:
            if self.first_address.raw_address == self.second_address.raw_address:
                is_identical = True

        return is_identical

    def comparison_report(self, nb_delimiters: Union[int, None] = None) -> None:
        """
        Print a formatted comparison report of the two addresses.
        """
        sys.stdout.writelines(self._comparison_report(nb_delimiters))

    def _comparison_report(self, nb_delimiters: Union[int, None]) -> str:
        """
        Builds a comparison_report with delimiters to make the beginning and the end of the comparison easier to spot.
        """

        # Get terminal size to adapt the output to the user
        nb_delimiters = os.get_terminal_size().columns if nb_delimiters is None else nb_delimiters

        formatted_str = ""
        comparison_report_signal = "=" * nb_delimiters
        formatted_str += comparison_report_signal + "\n"
        formatted_str += self._comparison_report_builder()
        formatted_str += comparison_report_signal + "\n\n"
        return formatted_str

    @abstractmethod
    def _comparison_report_builder(self) -> str:
        """
        Builds the core of a comparison report for the different comparisons. Since the procedure to make a tags
        comparison and the raw addresses comparison is different, the comparison report is not the same for the two.
        It is then implemented in each specific class.
        """

    @abstractmethod
    def _get_probs(self) -> Dict:
        """
        Get the tags from the parsing with their associated probabilities, the method needs to be implemented in each
        class because they don't use the probabilities the same way.
        """

    @staticmethod
    def _get_color_diff(string_one: str, string_two: str, highlight=False) -> str:
        """
        Compare two strings and determine the difference between the two. The differences are noted with colour code;
        if the first string has more element than the second one it will be noted in one colour; on the contrary,
        if the other string has something more, it will have a different colour notation.

        Args:
            string_one (str): The first string to compare.
            string_two (str): The second string to compare.
            highlight (bool, optional): If set to yes, the difference will be highlighted in colour instead of the
                character itself in colour. This might be used to have information where the discrepancies between
                two strings are spaces. The default is False.

        Notes:
            the method is colorblind-friendly, which means that the output will be
            in colours that minimize the risk that a user cannot see the difference as
            defined here https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40.

            If both the strings share the same character, it will be noted in white.
            If the first string has something more than the second one, it will be indicated in blue.
            If the second string has something more than the first one, it will be noted in yellow.

            The uses the SequenceMatcher to get the differences codes that are then converted to
            colour codes.

        Return:
            str: The two strings joined, and the differences are noted in colour codes
        """
        code_type = 48 if highlight else 38

        color_1 = "\033[{code_type};2;26;123;220m{text}\033[0m"  # blue
        color_2 = "\033[{code_type};2;255;194;10m{text}\033[0m"  # yellow

        white = "\033[38;2;255;255;255m{text}\033[0m"

        result = ""
        codes = SequenceMatcher(a=string_one, b=string_two).get_opcodes()
        for code in codes:
            if code[0] == "equal":
                result += white.format(text=(string_one[code[1]:code[2]]))
            elif code[0] == "delete":
                result += color_1.format(code_type=code_type, text=string_one[code[1]:code[2]])
            elif code[0] == "insert":
                result += color_2.format(code_type=code_type, text=string_two[code[3]:code[4]])
            elif code[0] == "replace":

                if code[1] <= code[3]:
                    result += (color_1.format(code_type=code_type, text=string_one[code[1]:code[2]]) +
                               color_2.format(code_type=code_type, text=string_two[code[3]:code[4]]))
                else:
                    result += (color_2.format(code_type=code_type, text=string_two[code[3]:code[4]]) +
                               color_1.format(code_type=code_type, text=string_one[code[1]:code[2]]))
        return result

    def _get_tags_diff_color(self,
                             name_one: str = "first address",
                             name_two: str = "second address",
                             verbose=True) -> str:
        """
        Print the output of the string with colour codes that represent the differences between the two strings.

        Args:
            name_one (str, optional) : Name associated with first color. The default value is the first address.
            name_two (str, optional) : Name associated with second color. The default value is the second address.
            verbose (bool, optional): If True, it will print a presentation of the colours and what they mean.
                The default value is True.

        """

        formatted_str = ""
        if verbose:
            formatted_str += "White: Shared\n"
            formatted_str += "Blue: Belongs only to the " + name_one + "\n"
            formatted_str += "Yellow: Belongs only to the " + name_two + "\n"
            formatted_str += "\n"

        address_component_names = [tag[0] for tag in self.list_of_bool if not tag[1]]

        for address_component_name in address_component_names:
            list_of_list_tag = []
            for parsed_address in [self.first_address.to_list_of_tuples(), self.second_address.to_list_of_tuples()]:
                list_of_list_tag.append(" ".join([tag for (tag, tag_name) in parsed_address \
                                                  if tag_name == address_component_name and tag is not None]))

            result = self._get_color_diff(list_of_list_tag[0], list_of_list_tag[1])

            formatted_str += address_component_name + ": \n"
            formatted_str += result + "\n"

        return formatted_str

    def _bool_address_tags_are_the_same(self, parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> List[tuple]:
        """
        Compare addresses components and put the differences in a dictionary where the keys are the
        names of the addresses components, and the values are the value of the addresses component.

        Args:
            parsed_addresses (Union[List[List[tuple]], List[tuple]]): Contains the tags and the
            address components name for the parsed addresses.

        Return:
            List[tuple]: List of tuples that contains all addresses components that differ from each other.
        """
        unique_address_component_names = self._unique_addresses_component_names(parsed_addresses)

        list_of_bool_and_tag = []
        for address_component_name in unique_address_component_names:
            list_of_list_tag = []
            for parsed_address in parsed_addresses:
                list_of_list_tag.append(" ".join([
                    tag for (tag, tag_name) in parsed_address if tag_name == address_component_name and tag is not None
                ]))

            list_of_bool_and_tag.append(
                (address_component_name, all(x == list_of_list_tag[0] for x in list_of_list_tag)))

        return list_of_bool_and_tag

    @staticmethod
    def _unique_addresses_component_names(parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> List:
        """
        Retrieves all the unique address components names from the comparison then returns it.

        Args:
            parsed_addresses (Union[List[List[tuple]], List[tuple]]): Contains the tags and the
            address components name for the parsed addresses.

        Return:
            Returns a list of all the unique address components names.
        """
        if isinstance(parsed_addresses[0], tuple):
            parsed_addresses = [parsed_addresses]

        # Here we don't use a set since order will change and report will also change.
        unique_address_component_names = []
        for tuple_values in parsed_addresses:
            for address_component in tuple_values:
                if isinstance(address_component[1], tuple):
                    address_component = address_component[1][0]
                else:
                    address_component = address_component[1]
                if address_component not in unique_address_component_names:
                    unique_address_component_names.append(address_component)

        return unique_address_component_names
