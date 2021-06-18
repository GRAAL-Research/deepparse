from deepparse.parser import AddressParser
from typing import List, Union


class AdressComparer:
    """
        Compares two addresses and return the the differences between them
    """
    
    def __init__(self, parser: AddressParser) -> None:
        """
        base address to make the comparisons
        """

        self.parser = parser

    
    def compare_with_deepparse(self, addresses_to_compare: Union[List[List[tuple]], List[tuple]]) -> dict:
        """
        305 rue des Lilas O, app 2 
        StreetNumber, StreetName ...
        [(305, StreetNumber), (rue, StreetName), ...]
        305 rue ...
        #{305: StreetNumber} risque de colission keys

        StreetName, StreetNumber

        """
        if isinstance(addresses_to_compare[0], tuple):  # when tag is also the tag and the probability of the tag
            addresses_to_compare = [addresses_to_compare]


        list_of_deepparsed_addresses = []
        for address in addresses_to_compare:
            rebuilt_raw_address = " ".join([element[0] for element in address])
            list_of_deepparsed_addresses.append(self.parser(rebuilt_raw_address))

        list_of_lists_of_tuple_of_deepparse_attr = []
        for deeppasred_address in list_of_deepparsed_addresses:
            list_of_lists_of_tuple_of_deepparse_attr.append(deeppasred_address.to_list_of_tuples()) 

        dict_of_delta_dicts = {}
        for index, address_to_compare in enumerate(addresses_to_compare):
            args_delta_dict = {'deepparse': list_of_lists_of_tuple_of_deepparse_attr[index],
                                str(index): address_to_compare}

            dict_of_delta_dicts[str(index)] = self.delta_dict_from_dict(args_delta_dict)

        return dict_of_delta_dicts


    def compare_raw_addresses(self, raw_addresses_to_compare: List[str]) -> dict:
        """
        305 rue des Lilas O, app 2 
        StreetNumber, StreetName ...
        [(305, StreetNumber), (rue, StreetName), ...]
        305 rue ...
        #{305: StreetNumber} risque de colission keys

        StreetName, StreetNumber

        """


        list_of_parsed_addresses = []
        for raw_address in raw_addresses_to_compare:
            list_of_parsed_addresses.append(self.parser(raw_address))

        args_delta_dict = {}
        for index, parsed_address in enumerate(list_of_parsed_addresses):
            args_delta_dict[str(index)] = parsed_address.to_list_of_tuples()
        
        delta_dict = self.delta_dict_from_dict(args_delta_dict)
        return delta_dict






    def delta_dict_from_dict(self, dict_of_parsed_addresses) -> dict:
        delta_dict = {}

        set_of_all_keys = set()
        for tuple_values in dict_of_parsed_addresses.values():
            for tag, key in tuple_values:
                set_of_all_keys.add(key)


        for key_iter in set_of_all_keys:
            dict_origin_string_tags = {}
            for origin, list_of_tag_and_key in dict_of_parsed_addresses.items():

                list_tag = [tag for (tag,key_tuple) in list_of_tag_and_key if key_tuple == key_iter and tag is not None]

                dict_origin_string_tags[origin] = " ".join(list_tag)


            if any (x != list(dict_origin_string_tags.values())[0] for x in dict_origin_string_tags.values()):
                delta_dict[key_iter] = dict_origin_string_tags


        return delta_dict


if __name__ == '__main__':

    list_of_tuples_address_one = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_address_two = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    raw_address_one = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_two = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_three = "325 rue des Lilas Ouest Québec Québec G1L 1B6"

    address_parser = AddressParser(model_type="bpemb", device=0)
    address_comparer = AdressComparer(address_parser)

    delta_dict_deeparse_one = address_comparer.compare_with_deepparse(list_of_tuples_address_one)
    delta_dict_deeparse_one_two = address_comparer.compare_with_deepparse([list_of_tuples_address_one, list_of_tuples_address_two])


    delta_dict_raw_addresses_one_two = address_comparer.compare_raw_addresses([raw_address_one, raw_address_two])
    delta_dict_raw_addresses_one_two_three = address_comparer.compare_raw_addresses([raw_address_one, raw_address_two, raw_address_three])


    delta_dict_from_dict = address_comparer.delta_dict_from_dict({'deeparse_one' :list_of_tuples_address_one,
                                'deeparse_two' :list_of_tuples_address_two})

    #test == parsed_address_same
    #test == parsed_address_diff


    #parsed_address_diff_streetNumber = address_parser("450 rue des Lilas Ouest Québec Québec G1L 1B6")
    #parsed_address_diff_streetName = address_parser("350 Boulevard des Lilas Ouest Québec Québec G1L 1B6")
    #parsed_address_diff_Unit = address_parser("350 rue des Lilas Ouest app 105 Québec Québec G1L 1B6")
    #parsed_address_diff_Municipality = address_parser("350 rue des Lilas Ouest Ste-Foy Québec G1L 1B6")
    #parsed_address_diff_Province = address_parser("350 rue des Lilas Ouest Québec Ontario G1L 1B6")
    #parsed_address_diff_PostalCode = address_parser("350 rue des Lilas Ouest Québec Québec G1P 1B6")
    #parsed_address_diff_Orientation = address_parser("350 rue des Lilas Est Québec Québec G1L 1B6")

    #dict_vide = test.delta_dict(parsed_address_same)
    #dict_numberStreet_diff = test.delta_dict(parsed_address_diff)

    #dict_parsed_address_diff_streetNumber = test.delta_dict(parsed_address_diff_streetNumber) 
    #dict_parsed_address_diff_streetName = test.delta_dict(parsed_address_diff_streetName) 
    #dict_parsed_address_diff_Unit = test.delta_dict(parsed_address_diff_Unit) 
    #dict_parsed_address_diff_Municipality = test.delta_dict(parsed_address_diff_Municipality) 
    #dict_parsed_address_diff_Province = test.delta_dict(parsed_address_diff_Province) 
    #dict_parsed_address_diff_PostalCode = test.delta_dict(parsed_address_diff_PostalCode) 
    #dict_parsed_address_diff_Orientation = test.delta_dict(parsed_address_diff_Orientation) 