from deepparse.parser import AddressParser
from typing import List, Union


class AdressComparer:
    """
        Compares addresses with each other and retrieves the differences between them.
    """
    
    def __init__(self, parser: AddressParser) -> None:
        """
        Address parser used to parse the addresses
        """

        self.parser = parser

    def __str__(self) -> str:
        return f"Compare addresses with {self.parser.model_type.capitalize()}AddressParser"

    __repr__ = __str__  # to call __str__ when list of address
    
    def compare_addresses_tags(self, addresses_to_compare: Union[List[List[tuple]], List[tuple]]) -> dict:
        """
        Compare a list of already parsed addresses with our results using our parser.

        Args:
            addresses_to_compare (Union[List[List[tuple]], List[tuple]]): List of tuple where
            the first element in the tuple is the value of the address component, and the 
            second element is the name of the address component

        Return:
            Dictionnary that contains dictionnaries that contains all addresses components that differ from the original
            parsing and the deepparsed components
        """
        if isinstance(addresses_to_compare[0], tuple):  # when tag is also the tag and the probability of the tag
            addresses_to_compare = [addresses_to_compare]

        #rebuilding addresses by joining the values of the address components
        #and then parse the addresses with deepparse
        list_of_deepparsed_addresses = []
        for address in addresses_to_compare:
            rebuilt_raw_address = " ".join([element[0] for element in address])
            list_of_deepparsed_addresses.append(self.parser(rebuilt_raw_address))


        #get the addresses components from deepparse
        list_of_lists_of_tuple_of_deepparse_attr = []
        for deepparsed_address in list_of_deepparsed_addresses:
            list_of_lists_of_tuple_of_deepparse_attr.append(deepparsed_address.to_list_of_tuples()) 

        #a dictionnary that will contains the delta dict of the parsed addresses to compare
        dict_of_delta_dicts = {}
        for index, address_to_compare in enumerate(addresses_to_compare):
            args_delta_dict = {'deepparse': list_of_lists_of_tuple_of_deepparse_attr[index],
                                str(index): address_to_compare}

            dict_of_delta_dicts[str(index)] = self.delta_dict_from_dict(args_delta_dict)

        return dict_of_delta_dicts


    def compare_raw_addresses(self, raw_addresses_to_compare: List[str]) -> dict:
        """
        Compare a list of raw addresses together, it starts by parsing the addresses
        with deepparse and then return the differences between the addresses components
        retrieved with deepparse.

        Args:
            raw_addresses_to_compare (List[str]): List of string that represent raw addresses.

        Return:
            Dictionnary that contains all addresses components that differ from each others
        """
        if len(raw_addresses_to_compare) < 2:
            raise ValueError("You need at least two addresses to compare") 

        #Parse addresses with deepparse
        list_of_parsed_addresses = []
        for raw_address in raw_addresses_to_compare:
            list_of_parsed_addresses.append(self.parser(raw_address))

        #Dict of all the deepparsed addresses components
        dict_of_parsed_addresses = {}
        for index, parsed_address in enumerate(list_of_parsed_addresses):
            dict_of_parsed_addresses[str(index)] = parsed_address.to_list_of_tuples()
        
        delta_dict = self.delta_dict_from_dict(dict_of_parsed_addresses)
        return delta_dict



    def delta_dict_from_dict(self, dict_of_parsed_addresses: dict) -> dict:
        """
        Compare addresses components and put the differences in a dict where the keys are the 
        names of the addresses components and the value are the value of the addresses components

        Args:
            dict_of_parsed_addresses (dict): The keys are the name of the parsed addresses and
            the values are the parsed address in the list of tupples format.

        Return:
            Dictionnary that contains all addresses components that differ from each others
        """

        delta_dict = {}

        #get all the unique addresses components
        set_of_all_keys = set()
        for tuple_values in dict_of_parsed_addresses.values():
            for tag, key in tuple_values:
                set_of_all_keys.add(key)

        #Iterate throught all the unique addresses components and retrieve the value
        #of the component for each parsed adresses
        for key_iter in set_of_all_keys:
            dict_origin_string_tags = {}
            for origin, list_of_tag_and_key in dict_of_parsed_addresses.items():
                
                #if there is more than one value per address component, the values
                #will be joined in a string.
                list_tag = [tag for (tag,key_tuple) in list_of_tag_and_key if key_tuple == key_iter and tag is not None]
                dict_origin_string_tags[origin] = " ".join(list_tag)

            #For each address components, if there is one value that differs from the rest,
            #the value of each parsed addresses with be added to the delta dict
            #where the key will be the address component name and the value will
            #be a dict that has the name of the parsed address as key and the
            # value of the address component as value. 
            if any (x != list(dict_origin_string_tags.values())[0] for x in dict_origin_string_tags.values()):
                delta_dict[key_iter] = dict_origin_string_tags


        return delta_dict


if __name__ == '__main__':

    list_of_tuples_address_one = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_address_two = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    raw_address_one = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_two = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_three = "355 rue des Lilas Ouest Québec Québec G1L 1B6"

    address_parser = AddressParser(model_type="bpemb", device=0)
    address_comparer = AdressComparer(address_parser)

    #Compare with deepparse
    delta_dict_deeparse_one = address_comparer.compare_with_deepparse(list_of_tuples_address_one)
    delta_dict_deeparse_one_two = address_comparer.compare_with_deepparse([list_of_tuples_address_one, list_of_tuples_address_two])

    #Compare raw addresses
    #Cant only compare one address
    delta_dict_raw_addresses_one = address_comparer.compare_raw_addresses([raw_address_one])

    #compare two addresses
    delta_dict_raw_addresses_one_two = address_comparer.compare_raw_addresses([raw_address_one, raw_address_two])

    #compare three addresses
    delta_dict_raw_addresses_one_two_three = address_comparer.compare_raw_addresses([raw_address_one, raw_address_two, raw_address_three])


    delta_dict_from_dict = address_comparer.delta_dict_from_dict({'deeparse_one' :list_of_tuples_address_one,
                                'deeparse_two' :list_of_tuples_address_two})

    
