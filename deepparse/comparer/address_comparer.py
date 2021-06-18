from deepparse.parser import AddressParser
from typing import List


class AdressComparer:
    """
        Compares two addresses and return the the differences between them
    """
    
    def __init__(self, parser: AddressParser) -> None:
        """
        base address to make the comparisons
        """

        self.parser = parser

    
    def deepparse_compare(self, address_to_compare: List[tuple]) -> dict:
        """
        305 rue des Lilas O, app 2 
        StreetNumber, StreetName ...
        [(305, StreetNumber), (rue, StreetName), ...]
        305 rue ...
        #{305: StreetNumber} risque de colission keys

        StreetName, StreetNumber

        """
        
        rebuilt_raw_address = " ".join([element[0] for element in address_to_compare])
        deepparsed_address = self.parser(rebuilt_raw_address)

        dict_of_deepparse_attr = deepparsed_address.to_dict()
        list_of_tuple_of_deepparse_attr = [(value, key) for key, value in dict_of_deepparse_attr.items()]

        args_delta_dict = {'name_list_one' : 'deepparse',
                            'name_list_two': 'compared',
                            'list_of_tuple_of_tags_one': list_of_tuple_of_deepparse_attr,
                            'list_of_tuple_of_tags_two': address_to_compare}

        delta_dict = self.delta_dict(**args_delta_dict)

        return delta_dict

    def raw_compare(self, raw_address_one: str, raw_address_two) -> dict:
        """
        305 rue des Lilas O, app 2 
        StreetNumber, StreetName ...
        [(305, StreetNumber), (rue, StreetName), ...]
        305 rue ...
        #{305: StreetNumber} risque de colission keys

        StreetName, StreetNumber

        """
        
        deepparsed_address_one = self.parser(raw_address_one)
        deepparsed_address_two = self.parser(raw_address_two)

        dict_of_deepparse_attr_one = deepparsed_address_one.to_dict()
        dict_of_deepparse_attr_two = deepparsed_address_two.to_dict()
        

        list_of_tuple_of_deepparse_attr_one = [(value, key) for key, value in dict_of_deepparse_attr_one.items()]
        list_of_tuple_of_deepparse_attr_two = [(value, key) for key, value in dict_of_deepparse_attr_two.items()]

        args_delta_dict = {'name_list_one' : 'raw_address_one',
                            'name_list_two': 'raw_address_two',
                            'list_of_tuple_of_tags_one': list_of_tuple_of_deepparse_attr_one,
                            'list_of_tuple_of_tags_two': list_of_tuple_of_deepparse_attr_two}

        delta_dict = self.delta_dict(**args_delta_dict)

        return delta_dict


    
    def delta_dict(self, name_list_one:str, name_list_two:str, 
                    list_of_tuple_of_tags_one: List[tuple], list_of_tuple_of_tags_two: List[tuple]) -> dict:
        delta_dict = {}

        list_of_keys_one = [element[1] for element in list_of_tuple_of_tags_one]
        list_of_keys_two = [element[1] for element in list_of_tuple_of_tags_two]

        set_of_all_keys= set(list_of_keys_one + list_of_keys_two)

        for key_iter in set_of_all_keys:
            list_tag_one = [tag for (tag,key_tuple) in list_of_tuple_of_tags_one if key_tuple == key_iter and tag is not None]
            list_tag_two = [tag for (tag,key_tuple) in list_of_tuple_of_tags_two if key_tuple == key_iter and tag is not None]

            tag_one = " ".join(list_tag_one) if list_tag_one else None
            tag_two = " ".join(list_tag_two) if list_tag_two else None


            if tag_one != tag_two:
                dict_diff = {name_list_one: tag_one, name_list_two: tag_two}
                delta_dict[key_iter] = dict_diff


        return delta_dict




if __name__ == '__main__':

    address_parser = AddressParser(model_type="bpemb", device=0)

    # you can parse one address
    #parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

    #parsed_address_same = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")
    #parsed_address_diff = address_parser("450 rue des Lilas Ouest Québec Québec G1L 1B6")


    address_comparer = AdressComparer(address_parser)
    delta_dict_deeparse = address_comparer.deepparse_compare([("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")])

    delta_dict_raw_addresses = address_comparer.raw_compare([("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")],
                                [("306", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")])

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

    a = {'a': 2, 'b' : 4}
    b = {'b': 4, 'a' : 2}
    a == b