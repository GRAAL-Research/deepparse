from typing import List, Union, Tuple
from difflib import Differ, SequenceMatcher
from pprint import pprint
import sys


class FormatedComparedAddress:

    def __init__(self, raw_addresses: Union[List[str], str],
                        parsed_tuples : List[List[Tuple]],
                        list_of_bool: List[Tuple[str, bool]],
                        type_of_comparison: str
                        ) -> None:
        """
        Address parser used to parse the addresses
        """
        self.raw_addresses = raw_addresses
        self.parsed_tuples = parsed_tuples
        self.list_of_bool = list_of_bool
        self.__type_of_comparison = type_of_comparison
        self.equivalent = self._equivalent()
        self.indentical = self._indentical()




    def __str__(self) -> str:
        return "Compared addresses"

    __repr__ = __str__  # to call __str__ when list of address

    def _equivalent(self) ->bool:
        return all([bool_address[1] for bool_address in self.list_of_bool])

    def _indentical(self) ->bool:
        is_identical = False
        if self._equivalent():
            if all(x == self.raw_addresses[0] for x in self.raw_addresses):
                is_identical = True
            
        return is_identical


    def print_tags_diff(self) -> None:
        if len(self.parsed_tuples) != 2:
            raise ValueError("Cannot compare more than two parsed adresses")

        address_component_names = [tag[0] for tag in self.list_of_bool if not tag[1]]
        
        for address_component_name in address_component_names:
            list_of_list_tag = []
            for parsed_address in self.parsed_tuples:

                #if there is more than one value per address component, the values
                #will be joined in a string.
                list_of_list_tag.append(" ".join([tag for (tag,tag_name) in parsed_address[0] if tag_name == address_component_name and tag is not None]))

    
            result = list(Differ().compare(list_of_list_tag[0], list_of_list_tag[1]))
            print(address_component_name + ": ")
            sys.stdout.writelines(result)
            print(" ")



    def print_tags_diff_color(self) -> None:
        if len(self.parsed_tuples) != 2:
            raise ValueError("Cannot compare more than two parsed adresses")

        address_component_names = [tag[0] for tag in self.list_of_bool if not tag[1]]
        
        for address_component_name in address_component_names:
            list_of_list_tag = []
            for parsed_address in self.parsed_tuples:

                #if there is more than one value per address component, the values
                #will be joined in a string.
                list_of_list_tag.append(" ".join([tag for (tag,tag_name) in parsed_address[0] if tag_name == address_component_name and tag is not None]))

    
            result = self.get_color_diff(list_of_list_tag[0], list_of_list_tag[1])
            print(address_component_name + ": ")
            sys.stdout.writelines(result)
            print(" ")
        

    def print_raw_diff(self) -> None:
        if len(self.raw_addresses) != 2:
            raise ValueError("Can only compare two adresses")
            
        
        result = list(Differ().compare(self.raw_addresses[0], self.raw_addresses[1]))
        pprint("Raw addresses: ")
        sys.stdout.writelines(result)
        print("")
    
    def print_raw_diff_color(self) -> None:
        if len(self.raw_addresses) != 2:
            raise ValueError("Can only compare two adresses")
            
        
        result = self.get_color_diff(self.raw_addresses[0], self.raw_addresses[1])
        print("Raw addresses: ")
        sys.stdout.writelines(result)
        print("")

    def get_probs(self):
        raw_address_prob = {}
        nb_raw_addresses = len(self.raw_addresses)
        for index, raw_address in enumerate(self.raw_addresses):
            raw_address_prob[raw_address] = self.parsed_tuples[2 - nb_raw_addresses + index][2]
            
        return raw_address_prob
        

        
    def get_color_diff(self, string_one, string_two):

        red = lambda text: f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"
        green = lambda text: f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"
        blue = lambda text: f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"
        white = lambda text: f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"


        result = ""
        codes = SequenceMatcher(a=string_one, b=string_two).get_opcodes()
        for code in codes:
            if code[0] == "equal": 
                result += white(string_one[code[1]:code[2]])
            elif code[0] == "delete":
                result += red(string_one[code[1]:code[2]])
            elif code[0] == "insert":
                result += green(string_two[code[3]:code[4]])
            elif code[0] == "replace":
                result += (red(string_one[code[1]:code[2]]) + green(string_two[code[3]:code[4]]))
        return result

    def _comparison_report_of_raw_addresses(self):
        if len(self.raw_addresses) < 2:
            raise ValueError("Must compare two raw addresses")
        print("-" * 50)

        intro_str = "Comparison report of the two raw addresses: "
        if self.indentical:
            print(intro_str +  "Identical")
        else:
            if self.equivalent:
                print(intro_str +  "Equivalent")
            else:
                print(intro_str +  "Not equivalent")
        print(" ")
        print("Address one: " + self.raw_addresses[0])
        print("and")
        print("Address two: " +self.raw_addresses[1])
        print(" ")


        print(" ")
        print("Probabilities of parsed tags for the addresses with " +self.parsed_tuples[0][1][1] +": ")
        print(" ")
        for index, tuple_dict in enumerate(self.get_probs().items()):
            key, value = tuple_dict
            print("Raw address: " + key)
            print(value)
            if index == 0:
                print(" ")

        if not self.equivalent:
            print(" ")
            print(" ")
            print("Addresses tags differences between the two addresses: ")
            print("Red: Address one has more")
            print("Green: Address two has more")
            print(" ")
            self.print_tags_diff_color()

        print("-" * 50)
        print(" ")

    def _comparison_report_of_tags(self):
        if len(self.raw_addresses) > 1:
            raise ValueError("Must compare two parsings for the same raw address")
        print("-" * 50)
        intro_str = "Comparison report of tags for parsed address: "
        if self.indentical:
            print(intro_str +"Identical")
        else:
            print(intro_str +"Not identical")
        print("Raw address: " + self.raw_addresses[0])

        print(" ")
        print("Tags: ")
        print(self.parsed_tuples[0][1] + ": ", self.parsed_tuples[0][0])
        print(" ")
        print(self.parsed_tuples[1][1][1] + ": ", self.parsed_tuples[1][0])
        print(" ")
        print(" ")
        print("Probabilities of parsed tags for the address:")
        print(" ")
        for index, tuple_dict in enumerate(self.get_probs().items()):
            key, value = tuple_dict
            print("Raw address: " + key)
            print(value)
            if index > 0:
                print(" ")

        if not self.indentical:
            print(" ")
            print(" ")
            print("Addresses tags differences between the two parsing:")
            print("Red: " + self.parsed_tuples[0][1] + " has more")
            print("Green: " + self.parsed_tuples[1][1][1] + " has more")
            print(" ")
            self.print_tags_diff_color()

        print("-" * 50)
        print(" ")

    def comparison_report(self):
        if self.__type_of_comparison == "raw":
            self._comparison_report_of_raw_addresses()
        elif self.__type_of_comparison == "tag":
            self._comparison_report_of_tags()



if __name__ == '__main__':

    

    
    list_of_tuples_address_one = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_address_two = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    raw_address_one = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
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









