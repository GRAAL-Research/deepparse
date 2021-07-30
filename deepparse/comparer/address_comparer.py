from typing import List, Union
from deepparse.comparer.formated_compared_address import FormatedComparedAddress
from deepparse.parser import AddressParser


# ça responsabilité est de comparer des adresses avec notre parsing d'adresse
# AdressComparer().compare([(addresse_1, [parsing]), (addresse_2, [parsing]), ..., (addresse_n, [parsing])])
# J'aimerai avoir à la sortie une liste de N objets d'adresse comparé
# L'objet outputer est comme un conteneur qui contient l'adresse original, la différence en liste de bool de la longueur
# du nombre de tag et une méthode __str__ pour afficher le output différent.
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
    
    def compare(self, addresses_to_compare: Union[List[tuple], List[List[tuple]]]) ->List[FormatedComparedAddress]:

        list_of_formated_compared_address = []
        for address_to_compare in addresses_to_compare:
            raw_address = address_to_compare[0]
            parsing_info = address_to_compare[1]
            list_of_bool_tuple = self.bool_address_tags_are_the_same([parsed_tuple[0] for parsed_tuple in parsing_info])

            list_of_formated_compared_address.append(FormatedComparedAddress(raw_address, parsing_info, list_of_bool_tuple))
        return list_of_formated_compared_address



    def compare_tags(self, addresses_to_compare: Union[List[tuple], List[List[tuple]]] ) ->List[FormatedComparedAddress]:
        """
        Compare a list of already parsed addresses with our results using our parser.

        Args:
            addresses_to_compare (Union[List[List[tuple]], List[tuple]]): List of addresses to parse represented in
            a lists of tuple where the first element in the tuple is the value of the address component, and the
            second element is the name of the address component.

        Return:
            Dictionnary that contains dictionnaries that contains all addresses components that differ from the original
            parsing and the deepparsed components
        """
        # donc ici, je vais vouloir
        # 1. parser les adresses avec notre approche
        # 2. Pour chaque adresse, comparer les tags (par token) entre eux
        # 3. Prendre cette liste de liste de booléen + les adresses originales, les parsings originaux et nos parsings
        # (un tuple de 3 ??)
        # et initiliazer un objet qui va avoir les attributs suivants (au moins)
        # la liste [(tag_1: bool), (tag_2: bool), ...]
        # l'adresse
        # le parsing original
        # notre parsing (avec les probs maybe???)
        # if perfectly identical or not
        if isinstance(addresses_to_compare[0], tuple):
            addresses_to_compare = [addresses_to_compare]


        rebuilt_raw_addresses = self.get_raw_addresses(addresses_to_compare)
        
        deepparsed_addresses = self.parser(rebuilt_raw_addresses)

        if not isinstance(deepparsed_addresses, list):
            deepparsed_addresses = [deepparsed_addresses]

        list_of_deepparse_tuple = []
        for parsed_address in deepparsed_addresses:
            dict_of_attr = parsed_address.to_dict()
            list_of_deepparse_tuple.append([(value, key) for key, value in dict_of_attr.items()])

        list_of_addresses_informations = [(raw_address, 
                                        [(address_to_compare, "source"), (deepparsed_tuple, "deepparse using " + self.parser.model_type.capitalize())]
                                        )
                                        for raw_address, address_to_compare, deepparsed_tuple
                                            in zip(rebuilt_raw_addresses, addresses_to_compare, list_of_deepparse_tuple)]

        return self.compare(list_of_addresses_informations)

    def compare_raw(self, list_of_addresses_to_compare: Union[List[str], List[List[str]]]) -> List[FormatedComparedAddress]:
        """
        Compare a list of raw addresses together, it starts by parsing the addresses
        with the setted parser and then return the differences between the addresses components
        retrieved with our model.

        Args:
            raw_addresses_to_compare (List[str]): List of string that represent raw addresses.

        Return:
            Dictionnary that contains all addresses components that differ from each others
        """
        if isinstance(list_of_addresses_to_compare[0], str):
            list_of_addresses_to_compare = [list_of_addresses_to_compare]

        list_of_addresses_informations = []
        for addresses_to_compare in list_of_addresses_to_compare:
            if len(addresses_to_compare) < 2:
                raise ValueError("You need at least two addresses to compare")
        
            deepparsed_addresses = self.parser(addresses_to_compare)


            list_of_deepparse_tuple = []
            for parsed_address in deepparsed_addresses:
                dict_of_attr = parsed_address.to_dict()
                list_of_deepparse_tuple.append([(value, key) for key, value in dict_of_attr.items()])

            list_of_addresses_informations.append((addresses_to_compare, 
                                        [(deepparsed_tuple, repr(deepparsed_address))
                                            for deepparsed_tuple, deepparsed_address
                                            in zip(list_of_deepparse_tuple, deepparsed_addresses)]))
        
        return self.compare(list_of_addresses_informations)




    def addresses_component_names(self, parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> set:
        if isinstance(parsed_addresses[0], tuple):
            parsed_addresses = [parsed_addresses]

        set_of_all_address_component_names = set()
        for tuple_values in parsed_addresses:
            for address_component in tuple_values:
                set_of_all_address_component_names.add(address_component[1])

        return set_of_all_address_component_names

    def get_raw_addresses(self, parsed_addresses):
            
        if not isinstance(parsed_addresses[0], list):
            parsed_addresses = [parsed_addresses]

        return  [ " ".join([element[0] for element in address]) for address in parsed_addresses]
            
    def bool_address_tags_are_the_same(self, parsed_addresses: Union[List[List[tuple]], List[tuple]]) -> List[tuple]:
        """
        Compare addresses components and put the differences in a dict where the keys are the
        names of the addresses components and the value are the value of the addresses components

        Return:
            Dictionnary that contains all addresses components that differ from each others
        """



        list_of_bool_and_tag = []

        #get all the unique addresses components
        set_of_all_address_component_names = self.addresses_component_names(parsed_addresses)


        #Iterate throught all the unique addresses components and retrieve the value
        #of the component for each parsed adresses
        for address_component_name in set_of_all_address_component_names:
            list_of_list_tag = []
            for parsed_address in parsed_addresses:

                #if there is more than one value per address component, the values
                #will be joined in a string.
                list_of_list_tag.append(" ".join([tag for (tag,tag_name) in parsed_address if tag_name == address_component_name and tag is not None]))

                #For each address components, if there is one value that differs from the rest,
                #the value of each parsed addresses with be added to the delta dict
                #where the key will be the address component name and the value will
                #be a dict that has the name of the parsed address as key and the
                # value of the address component as value.
            list_of_bool_and_tag.append((address_component_name, all(x == list_of_list_tag[0] for x in list_of_list_tag) ))

        return list_of_bool_and_tag

if __name__ == '__main__':


    
    list_of_tuples_address_one = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_address_two = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    raw_address_one = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_two = "305 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_three = "355 rue des Lilas Ouest Québec Québec G1L 1B6"

    #test = list(d.compare(raw_address_one, raw_address_three))
    #pprint(test)
    

    address_parser = AddressParser(model_type="bpemb", device=0)
    address_comparer = AdressComparer(address_parser)

    #Compare with source tags with deepparse tags
    delta_dict_deeparse_one = address_comparer.compare_tags(list_of_tuples_address_one)
    delta_dict_deeparse_one_two = address_comparer.compare_tags([list_of_tuples_address_one, list_of_tuples_address_two])
    delta_dict_deeparse_one_two[0].print_tags_diff()
    delta_dict_deeparse_one_two[0].print_tags_diff_color()

    delta_dict_deeparse_one_two[0].comparison_report()

    #compare two equivalent addresses
    delta_dict_raw_addresses_one_two = address_comparer.compare_raw([raw_address_one, raw_address_two])
    delta_dict_raw_addresses_one_two[0].comparison_report()
    #delta_dict_raw_addresses_one_two[0].print_raw_diff_color()
    #delta_dict_raw_addresses_one_two[0].print_tags_diff()
    #delta_dict_raw_addresses_one_two[0].raw_addresses

    #compare two not equivalent addresses
    delta_dict_raw_addresses_one_three = address_comparer.compare_raw([raw_address_one, raw_address_three])
    delta_dict_raw_addresses_one_three[0].comparison_report()

    #compare three addresses
    #delta_dict_raw_addresses_one_two_three = address_comparer.compare_raw([[raw_address_one, raw_address_two], [raw_address_one, raw_address_two, raw_address_three]])
    #print(delta_dict_raw_addresses_one_two_three[1].equivalent)

