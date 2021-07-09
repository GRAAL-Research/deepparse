from typing import List, Union, Tuple
from difflib import Differ, SequenceMatcher
from pprint import pprint
import sys



class FormatedComparedAddress:

    def __init__(self, raw_address: Union[List[str], str],
                        parsed_tuples : List[List[Tuple]],
                        list_of_bool: List[Tuple[str, bool]]) -> None:
        """
        Address parser used to parse the addresses
        """
        self.raw_address = raw_address
        self.parsed_tuples = parsed_tuples
        self.list_of_bool = list_of_bool
        self.equivalent = self._equivalent()




    def __str__(self) -> str:
        return "Compared addresses"

    __repr__ = __str__  # to call __str__ when list of address

    def _equivalent(self) ->bool:
        return all([bool_address[1] for bool_address in self.list_of_bool])

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
            pprint(address_component_name + " : ")
            sys.stdout.writelines(result)
            print("")


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
            pprint(address_component_name + " : ")
            sys.stdout.writelines(result)
            print("")
        

    def print_raw_diff(self) -> None:
        if len(self.raw_address) != 2:
            raise ValueError("Can only compare two adresses")
            
        
        result = list(Differ().compare(self.raw_address[0], self.raw_address[1]))
        pprint("Raw addresses : ")
        sys.stdout.writelines(result)
        print("")
    
    def print_raw_diff_color(self) -> None:
        if len(self.raw_address) != 2:
            raise ValueError("Can only compare two adresses")
            
        
        result = self.get_color_diff(self.raw_address[0], self.raw_address[1])
        pprint("Raw addresses : ")
        sys.stdout.writelines(result)
        print("")

        
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









