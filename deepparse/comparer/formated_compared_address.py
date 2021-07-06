from typing import List, Union

class FormatedComparedAddress:

    def __init__(self, list_of_parsed_address_tuple: Union[List[List[tuple]], List[tuple]], parser):
        """
        Address parser used to parse the addresses
        """
        self.parser = parser
        self.list_of_parsed_address_tuple = list_of_parsed_address_tuple
        self.delta_dict = self.address_diff()

    def __str__(self) -> str:
        return f"Compared addresses with {self.parser.model_type.capitalize()}AddressParser"

    __repr__ = __str__  # to call __str__ when list of address

    def address_diff(self) -> dict:
        """
        Compare addresses components and put the differences in a dict where the keys are the
        names of the addresses components and the value are the value of the addresses components

        Return:
            Dictionnary that contains all addresses components that differ from each others
        """
        delta_dict = {}

        #get all the unique addresses components
        for tuple_values in self.list_of_parsed_address_tuple:
            set_of_all_address_component_names = set()
            for address_component in tuple_values[0]:
                set_of_all_address_component_names.add(address_component[1])


            #Iterate throught all the unique addresses components and retrieve the value
            #of the component for each parsed adresses
            for address_component_name in set_of_all_address_component_names:
                dict_origin_string_tags = {}

                #if there is more than one value per address component, the values
                #will be joined in a string.
                list_tag = [tag for (tag,key_tuple) in tuple_values[0] if key_tuple == address_component_name and tag is not None]
                dict_origin_string_tags[tuple_values[1]] = " ".join(list_tag)

                #For each address components, if there is one value that differs from the rest,
                #the value of each parsed addresses with be added to the delta dict
                #where the key will be the address component name and the value will
                #be a dict that has the name of the parsed address as key and the
                # value of the address component as value.
                if any (x != list(dict_origin_string_tags.values())[0] for x in dict_origin_string_tags.values()):
                    delta_dict[address_component_name] = dict_origin_string_tags


        return delta_dict

    def get_diff(self):
        return self.delta_dict