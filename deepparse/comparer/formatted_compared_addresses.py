import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Union, Tuple

from ..parser import FormattedParsedAddress


@dataclass
class FormattedComparedAddresses(ABC):
    """
    A comparison for addresses returned by the address comparer

    Args:
        address_one(FormattedParsedAddress) : A FormatedParsedAddress that contains the parsing information
                                                for the first address
        address_two(FormattedParsedAddress) : A FormatedParsedAddress that contains the parsing information
                                                for the second address
        origin: (Tuple[str]) : the origin of the parsing (ex : from source or from deepparse)

        colorblind (bool) :  a colorbind flag, weither the use wants a colorblind friendly output or not

    Attributes:
        address_one(FormattedParsedAddress) : A FormatedParsedAddress that contains the parsing information
                                                for the first address
        address_two(FormattedParsedAddress) : A FormatedParsedAddress that contains the parsing information
                                                for the second address
        origin: (Tuple[str]) : the origin of the parsing (ex : from source or from deepparse)

        colorblind (bool) :  a colorbind flag, weither the use wants a colorblind friendly output or not

        list_of_bool (List[Tuple]): list_of_bool that contains all the address components name and indicates if it
                                is the same for the two addresses

    Example:

        .. code-block:: python

            address_comparer = AdressComparer(AddressParser())
            raw_identical_comparison = address_comparer.compare_raw(
                                                        ("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6",
                                                        "450 rue des Lilas Ouest Quebec city Quebec G1L 1B6"))

            print(raw_identical_comparison.raw_addresses) # [350 rue des Lilas Ouest Quebec city Quebec G1L 1B6,
                                                            450 rue des Lilas Ouest Quebec city Quebec G1L 1B6]

            print(raw_identical_comparison.address_parsed_components)
            #[[('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'),
            #  ('Ouest Quebec city', 'Municipality'),
            #   ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')],
            #[('450', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'),
            # ('Ouest Quebec city', 'Municipality'),
            #  ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')]]

    """
    address_one: FormattedParsedAddress
    address_two: FormattedParsedAddress
    origin: Tuple[str]
    colorblind: bool

    def __post_init__(self) -> None:
        self.list_of_bool = self._bool_address_tags_are_the_same([self.address_one.to_list_of_tuples(),
                                                                  self.address_two.to_list_of_tuples()])

    @property
    def equivalent(self) -> bool:
        """Check if the parsing is the same for the two addresses

        Returns:
            bool: if the parsing is the same for the two addresses
        """
        return all([bool_address[1] for bool_address in self.list_of_bool])

    @property
    def identical(self) -> bool:
        """Check if the parsing is the same for the two addresses and if
            the raw addresses are identical

        Returns:
            bool: if the parsing is the same for the two addresses and if
            the raw addresses are identical
        """
        is_identical = False
        if self.equivalent:
            if self.address_one.raw_address == self.address_two.raw_address:
                is_identical = True

        return is_identical

    @abstractmethod
    def comparison_report(self) -> None:
        """print a comparison report for the different comparisons, since the procedure in order
            to make a tags comparison and the raw addresses comparison is different, the comparison
            report are not the same for the two. It is then implemented in each specific classes.
        """

    @abstractmethod
    def get_probs(self):
        """get the tags from the parsin with their associated probabilities, the method
            needs to be implemented in each class because they dont use the probabilities
            the same way.

        Returns:
            Dict: A dict where the keys are the raw addresses and the values are
                    the tags with their associated probabilities
        """

    def _get_color_diff(self, string_one: str, string_two: str, highlight=False) -> str:
        """compare two string and determine the difference between the two.
            the differences are noted with color code, if the first string has
            more element than the second one it will be noted in one color, but
            on the contrary if the other string has something more it will have
            a different color notation.

        Args:
            string_one (str): the first string to compare
            string_two (str): the second string to compare
            highlight (bool, optional): if set to yes, the difference will be highlighted in color
                                        instead of the character itself in color. This might be used
                                        to have information where the differences among two strings are
                                        spaces. Defaults to False.

        Notes:
            the method is colorblind friendly, if the colorblind flag is raised,
            the output will be in colors that minimize the risk that a user cannot
            see the difference.

            If both the strings share the same charachter, it will be noted in white.
            If the first string has something more than the second one, it will be noted in
            red (or in blue for the colorblind mode).
            If the second string has something more than the first one, it will be noted in
            green (or in yellow for the colorblind mode).

            The uses the SequenceMatcher to get the differences codes that are then converted to
            color codes.

        Returns:
            str: the two strings joined and the differences are noted in color codes
        """

        code_type = 48 if highlight else 38

        if self.colorblind:
            # https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
            # pas une bonne pratique de sauvegarder des lambdas, en fait c'est plutôt juste du
            # formatting de string de la string f"\033[{code_type};2;26;123;220m{text}\033[0m"
            # donc je ferai de quoi du genre
            # f"\033[{code_type};2;26;123;220m{}\033[0m"
            # et plus tard color_1.format(text)

            color_1 = lambda text: f"\033[{code_type};2;26;123;220m{text}\033[0m"  # blue
            color_2 = lambda text: f"\033[{code_type};2;255;194;10m{text}\033[0m"  # yellow
        else:
            color_1 = lambda text: f"\033[{code_type};2;255;0;0m{text}\033[0m"  # red
            color_2 = lambda text: f"\033[{code_type};2;0;255;0m{text}\033[0m"  # green

        white = lambda text: f"\033[38;2;255;255;255m{text}\033[0m"

        result = ""
        codes = SequenceMatcher(a=string_one, b=string_two).get_opcodes()
        for code in codes:
            if code[0] == "equal":
                result += white(string_one[code[1]:code[2]])
            elif code[0] == "delete":
                result += color_1(string_one[code[1]:code[2]])
            elif code[0] == "insert":
                result += color_2(string_two[code[3]:code[4]])
            elif code[0] == "replace":

                if code[1] <= code[3]:
                    result += (color_1(string_one[code[1]:code[2]]) + color_2(string_two[code[3]:code[4]]))
                else:
                    result += (color_2(string_two[code[3]:code[4]]) + color_1(string_one[code[1]:code[2]]))
        return result

    def _print_probs_of_tags(self, verbose=True) -> None:
        """takes the tags and their probabilities and print them to console

        Args:
            verbose (bool, optional): If true, the results are presented. Defaults to True.
        """
        if verbose:
            print("Probabilities of parsed tags for the address:")
            print("")
        for index, tuple_dict in enumerate(self.get_probs().items()):
            key, value = tuple_dict
            print("Raw address: " + key)
            print(value)
            if index > 0:
                print("")

    def _print_tags_diff_color(self, verbose=True) -> None:
        """Print the output of the string with color codes that represent
        the differences among the two strings.

        Args:
            verbose (bool, optional): If True, it will print a presentation of the colors
            and what they mean. Defaults to True.
        """
        if verbose:
            print("White: Shared")
            if not self.colorblind:
                print("Red: Belongs only to " + self.origin[0])
                print("Green: Belongs only to " + self.origin[1])
            else:
                print("Blue: Belongs only to " + self.origin[0])
                print("Yellow: Belongs only to " + self.origin[1])
            print("")

        address_component_names = [tag[0] for tag in self.list_of_bool if not tag[1]]

        for address_component_name in address_component_names:
            list_of_list_tag = []
            for parsed_address in [self.address_one.to_list_of_tuples(), self.address_two.to_list_of_tuples()]:
                # if there is more than one value per address component, the values
                # will be joined in a string.
                list_of_list_tag.append(" ".join([tag for (tag, tag_name) in parsed_address \
                                                  if tag_name == address_component_name and tag is not None]))

            result = self._get_color_diff(list_of_list_tag[0], list_of_list_tag[1])

            print(address_component_name + ": ")
            sys.stdout.writelines(result)
            print("")

    def _bool_address_tags_are_the_same(self, parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> List[tuple]:
        """
        Compare addresses components and put the differences in a dict where the keys are the
        names of the addresses components and the value are the value of the addresses component

        Args:
            parsed_addresses (Union[List[List[tuple]], List[tuple]]): Contains the tags and the
            address components name for the parsed addresses.

        Returns:
            List[tuple]: List of tuples that contains all addresses components that differ from each others
        """
        set_of_all_address_component_names = self._addresses_component_names(parsed_addresses)

        list_of_bool_and_tag = []
        for address_component_name in set_of_all_address_component_names:
            list_of_list_tag = []
            for parsed_address in parsed_addresses:
                list_of_list_tag.append(" ".join([tag for (tag, tag_name) in parsed_address if
                                                  tag_name == address_component_name and tag is not None]))

            list_of_bool_and_tag.append(
                (address_component_name, all(x == list_of_list_tag[0] for x in list_of_list_tag)))

        return list_of_bool_and_tag

    # C'est une méthode statique
    @staticmethod
    def _addresses_component_names(parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> set:
        """Retrieves all the unique address components names from the comparison then returns it.

        Args:
            parsed_addresses (Union[List[List[tuple]], List[tuple]]): Contains the tags and the
            address components name for the parsed addresses.

        Returns:
            set: returns a set of all the unique address components names
        """
        if isinstance(parsed_addresses[0], tuple):
            parsed_addresses = [parsed_addresses]

        set_of_all_address_component_names = set()
        for tuple_values in parsed_addresses:
            for address_component in tuple_values:

                if isinstance(address_component[1], tuple):
                    address_component = address_component[1][0]
                else:
                    address_component = address_component[1]
                set_of_all_address_component_names.add(address_component)

        return set_of_all_address_component_names
