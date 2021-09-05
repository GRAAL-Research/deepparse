from typing import List, Union, Dict
from difflib import  SequenceMatcher
import sys
import os


class FormatedComparedAddress:
    """
    A comparison for addresses returned by the address comparer

    Args:
        addresses (Union[Dict, List[Dict]]): A dictionnary where the keys are the name of
        the addresses components and the values contain the information for this specific
        component.
    
        colorblind (bool, optional): A flag that will print the comparison report in
        colorblind friendly colors if set to True. Defaults to False.

    Attributes:
        raw_addresses: The raw addresses (not parsed)
        address_parsed_components: The parsed address in a list of tuples where the first elements
            are the address components and the second elements are the tags.

    Example:

        .. code-block:: python

            address_comparer = AdressComparer(AddressParser())
            raw_identical_comparison = address_comparer.compare_raw(("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6",
                                                                    "450 rue des Lilas Ouest Quebec city Quebec G1L 1B6"))
            
            print(raw_identical_comparison.raw_addresses) # [350 rue des Lilas Ouest Quebec city Quebec G1L 1B6,
                                                            450 rue des Lilas Ouest Quebec city Quebec G1L 1B6]

            print(raw_identical_comparison.address_parsed_components)
            #[[('350', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Ouest Quebec city', 'Municipality'),
            #   ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')],
            #[('450', 'StreetNumber'), ('rue des Lilas', 'StreetName'), (None, 'Unit'), ('Ouest Quebec city', 'Municipality'),
            #  ('Quebec', 'Province'), ('G1L 1B6', 'PostalCode'), (None, 'Orientation'), (None, 'GeneralDelivery')]]

    """
    def __init__(self, addresses:Dict,
                        colorblind:bool = False) -> None:

        self.raw_addresses = addresses["raw_addresses"]
        self.address_parsed_components = [addresses["address_one"]["tags"], addresses["address_two"]["tags"]]
        
        self.__list_of_bool = self._bool_address_tags_are_the_same(self.address_parsed_components)
        self.__type_of_comparison = addresses["type_of_comparison"]
        self.__address_dict = addresses
        self.__colorblind = colorblind


    def __str__(self) -> str:
        return "Compared addresses"

    __repr__ = __str__  # to call __str__ when list of address

    @property
    def equivalent(self) ->bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return all([bool_address[1] for bool_address in self.__list_of_bool])

    @property
    def indentical(self) ->bool:
        """[summary]

        Returns:
            bool: [description]
        """
        is_identical = False
        if self.equivalent:
            if all(x == self.raw_addresses[0] for x in self.raw_addresses):
                is_identical = True

        return is_identical



    def _print_raw_diff_color(self, address_one_name: str, address_two_name:str, verbose = True) -> None:
        """Print the raw addresses and highlight the differences between them.

        Raises:
            ValueError: Can only use the method if the comparison is for raw addresses, meaning at least
            two raw addresses has been passed in arguments.
        """
        if len(self.raw_addresses) != 2:
            raise ValueError("Can only compare two adresses")

        result = self._get_color_diff(self.raw_addresses[0], self.raw_addresses[1], highlight=True)

        if verbose:
            print("White: Shared")
            if not self.__colorblind:
                print("Red: Belongs only to " + address_one_name)
                print("Green: Belongs only to " + address_two_name)
            else:
                print("Blue: Belongs only to " + address_one_name)
                print("Yellow: Belongs only to " + address_two_name)
            print("")
        sys.stdout.writelines(result)
        print("")


    def _print_tags_diff_color(self, address_one_name: str, address_two_name:str, verbose = True) -> None:
        """Print the addresses tags and highlight the differences between them.

        Raises:
            ValueError: Can only use the method if the comparison is for raw addresses, meaning at least
            two raw addresses has been passed in arguments.
        """
        if len(self.address_parsed_components) != 2:
            raise ValueError("Can only compare two adresses")

        if verbose:
            print("White: Shared")
            if not self.__colorblind:
                print("Red: Belongs only to " + address_one_name)
                print("Green: Belongs only to " + address_two_name)
            else:
                print("Blue: Belongs only to " + address_one_name)
                print("Yellow: Belongs only to " + address_two_name)
            print("")
        address_component_names = [tag[0] for tag in self.__list_of_bool if not tag[1]]

        for address_component_name in address_component_names:
            list_of_list_tag = []
            for parsed_address in self.address_parsed_components:

                #if there is more than one value per address component, the values
                #will be joined in a string.
                list_of_list_tag.append(" ".join([tag for (tag,tag_name) in parsed_address \
                if tag_name == address_component_name and tag is not None]))

            result = self._get_color_diff(list_of_list_tag[0], list_of_list_tag[1])

                
            print(address_component_name + ": ")
            sys.stdout.writelines(result)
            print("")




    def get_probs(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        raw_address_prob = {}
        nb_raw_addresses = len(self.raw_addresses)
        if nb_raw_addresses == 1 :
            raw_address_prob[self.raw_addresses[0]] = self.__address_dict["address_two"]["probs"]
        else:
            raw_address_prob[self.raw_addresses[0]] = self.__address_dict["address_one"]["probs"]
            raw_address_prob[self.raw_addresses[1]] = self.__address_dict["address_two"]["probs"]

        return raw_address_prob

    def _get_color_diff(self, string_one, string_two, highlight = False):
        """[summary]

        Args:
            string_one ([type]): [description]
            string_two ([type]): [description]

        Returns:
            [type]: [description]
        """


        
        code_type = 48 if highlight else 38


        if self.__colorblind:
            #https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
            color_1 = lambda text: f"\033[{code_type};2;26;123;220m{text}\033[0m" #blue
            color_2 = lambda text: f"\033[{code_type};2;255;194;10m{text}\033[0m" #yellow
        else:
            color_1 = lambda text: f"\033[{code_type};2;255;0;0m{text}\033[0m" #red
            color_2 = lambda text: f"\033[{code_type};2;0;255;0m{text}\033[0m" #green


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

    def _comparison_report_of_raw_addresses(self):
        """[summary]

        Raises:
            ValueError: [description]
        """
        if len(self.raw_addresses) < 2:
            raise ValueError("Must compare two raw addresses")


        intro_str = "Comparison report of the two raw addresses: "
        if self.indentical:
            print(intro_str +  "Identical")
        else:
            if self.equivalent:
                print(intro_str +  "Equivalent")
            else:
                print(intro_str +  "Not equivalent")
        print("")
        print("Address one: " + self.raw_addresses[0])
        print("and")
        print("Address two: " +self.raw_addresses[1])
        print("")



        print("")
        print("Probabilities of parsed tags for the addresses with "+self.__address_dict["address_one"]["origin"] +": ")
        print("")
        probs = list(self.get_probs().values())
        print("Parsed address: "+ self.__address_dict["address_one"]["repr"])
        print(probs[0])
        if not self.indentical:
            print("")
            print("Parsed address: "+ self.__address_dict["address_two"]["repr"])
            print(probs[1])

            if self.equivalent:
                print("")
                print("")
                print("Raw differences between the two addresses: ")
                self._print_raw_diff_color("Address one", "Address two")
            else:
                print("")
                print("")
                print("Addresses tags differences between the two addresses: ")
                self._print_tags_diff_color("Address one", "Address two")

    def _comparison_report_of_tags(self):
        """[summary]

        Raises:
            ValueError: [description]
        """
        if len(self.raw_addresses) > 1:
            raise ValueError("Must compare two parsings for the same raw address")

        intro_str = "Comparison report of tags for parsed address: "
        if self.indentical:
            print(intro_str +"Identical")
        else:
            print(intro_str +"Not identical")
        print("Raw address: " + self.raw_addresses[0])

        print("")
        print("Tags: ")
        print(self.__address_dict["address_one"]["origin"] + ": ", self.address_parsed_components[0])
        print("")
        print(self.__address_dict["address_two"]["origin"]  + ": ", self.address_parsed_components[1])
        print("")
        print("")
        
        self._print_probs_of_tags()

        if not self.indentical:
            print("")
            print("")
            print("Addresses tags differences between the two parsing:")
            self._print_tags_diff_color(self.__address_dict["address_one"]["origin"], self.__address_dict["address_two"]["origin"])
        print("")


    def comparison_report(self) -> None:
        """[summary]

        Args:
            nb_delimiters (int, optional): [description]. Defaults to None.
        """
        #get terminal size to adapt the output to the user
        #nb_delimiters = os.get_terminal_size().columns if nb_delimiters is None else nb_delimiters
        nb_delimiters = 125

        comparison_report_signal = "=" * nb_delimiters
        print(comparison_report_signal)
        if self.__type_of_comparison == "raw":
            self._comparison_report_of_raw_addresses()
        elif self.__type_of_comparison == "tag":
            self._comparison_report_of_tags()
        print(comparison_report_signal)
        print("")


    def _print_probs_of_tags(self, verbose = True) -> None:
        if verbose:
            print("Probabilities of parsed tags for the address:")
            print("")
        for index, tuple_dict in enumerate(self.get_probs().items()):
            key, value = tuple_dict
            print("Raw address: " + key)
            print(value)
            if index > 0:
                print("")
        

    def _bool_address_tags_are_the_same(self, parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> List[tuple]:
        """
        Compare addresses components and put the differences in a dict where the keys are the
        names of the addresses components and the value are the value of the addresses components

        Return:
            List of tuples that contains all addresses components that differ from each others
        """

        __list_of_bool_and_tag = []

        # get all the unique addresses components
        set_of_all_address_component_names = self._addresses_component_names(parsed_addresses)

        # Iterate throught all the unique addresses components and retrieve the value
        # of the component for each parsed adresses
        for address_component_name in set_of_all_address_component_names:
            list_of_list_tag = []
            for parsed_address in parsed_addresses:
                # if there is more than one value per address component, the values
                # will be joined in a string.
                list_of_list_tag.append(" ".join([tag for (tag, tag_name) in parsed_address if
                                                tag_name == address_component_name and tag is not None]))

                # For each address components, if there is one value that differs from the rest,
                # the value of each parsed addresses with be added to the delta dict
                # where the key will be the address component name and the value will
                # be a dict that has the name of the parsed address as key and the
                # value of the address component as value.
            __list_of_bool_and_tag.append(
                (address_component_name, all(x == list_of_list_tag[0] for x in list_of_list_tag)))

        return __list_of_bool_and_tag

    def _addresses_component_names(self, parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> set:
        """[summary]

        Args:
            parsed_addresses (Union[List[List[tuple]], List[tuple]]): [description]

        Returns:
            set: [description]
        """
        if isinstance(parsed_addresses[0], tuple):
            parsed_addresses = [parsed_addresses]

        set_of_all_address_component_names = set()
        for tuple_values in parsed_addresses:
            for address_component in tuple_values:
                set_of_all_address_component_names.add(address_component[1])

        return set_of_all_address_component_names




if __name__ == '__main__':
    list_of_tuples_address_one = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_address_two = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    raw_address_one = "305  rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_two = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_three = "355 rue des Chemins Ouest Québec Québec G1L 1B6"
    #result = list(Differ().compare(raw_address_one, raw_address_three))
    #
    #def test():
    #    pprint("Raw addresses : ")
    #    sys.stdout.writelines(result)
    #    print("")
    #
    #test()
