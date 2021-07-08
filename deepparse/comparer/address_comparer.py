from deepparse.comparer.formated_compared_address import FormatedComparedAddress
from deepparse.parser import AddressParser
from typing import List, Union


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
    
    def compare_addresses_tags(self, addresses_to_compare: Union[List[List[tuple]], List[tuple]]) -> List:
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

        #rebuilding addresses by joining the values of the address components
        #and then parse the addresses with deepparse
        

        rebuilt_raw_address = [ " ".join([element[0] for element in address]) for address in addresses_to_compare]
        
        deepparsed_addresses = self.parser(rebuilt_raw_address)

        if not isinstance(deepparsed_addresses, list):
            deepparsed_addresses = [deepparsed_addresses]

        list_of_list_of_tuple = []
        for parsed_address in deepparsed_addresses:
            dict_of_attr = parsed_address.to_dict()
            list_of_list_of_tuple.append([(value, key) for key, value in dict_of_attr.items()]) 

        list_of_source_tuple = [(formated_address, 'source') for formated_address in addresses_to_compare]
        list_of_deepparsed_tuple = [(formated_address, 'deepparse') for formated_address in list_of_list_of_tuple]

        diff_tuple = []
        for already_parsed_address, deepparsed_address, raw_address in zip(list_of_source_tuple, list_of_deepparsed_tuple, rebuilt_raw_address):
            diff_tuple.append([FormatedComparedAddress([already_parsed_address, deepparsed_address], self.parser), (raw_address, already_parsed_address, deepparsed_address)])
        
        return diff_tuple


    def compare_raw_addresses(self, raw_addresses_to_compare: Union[List[str], List[List[str]]]) -> dict:
        """
        Compare a list of raw addresses together, it starts by parsing the addresses
        with the setted parser and then return the differences between the addresses components
        retrieved with our model.

        Args:
            raw_addresses_to_compare (List[str]): List of string that represent raw addresses.

        Return:
            Dictionnary that contains all addresses components that differ from each others
        """
        if isinstance(raw_addresses_to_compare[0], str):
            raw_addresses_to_compare = [raw_addresses_to_compare]

        diff_tuple = []
        for addresses_to_compare in raw_addresses_to_compare:
            if len(addresses_to_compare) < 2:
                raise ValueError("You need at least two addresses to compare")

            #Parse addresses with deepparse
            deepparsed_addresses = self.parser(addresses_to_compare)

            list_of_list_of_tuple = []
            for parsed_address in deepparsed_addresses:

                list_of_list_of_tuple.append(([(value, key) for key, value in parsed_address.to_dict().items()],'deepparse'))
                
            diff_tuple.append([FormatedComparedAddress(list_of_list_of_tuple, self.parser), zip(addresses_to_compare, list_of_list_of_tuple)])
        

        
        return diff_tuple


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
    delta_dict_deeparse_one = address_comparer.compare_addresses_tags(list_of_tuples_address_one)
    delta_dict_deeparse_one_two = address_comparer.compare_addresses_tags([list_of_tuples_address_one, list_of_tuples_address_two])

    test = delta_dict_deeparse_one_two[0][1]



    #Compare raw addresses
    #Cant only compare one address
    #delta_dict_raw_addresses_one = address_comparer.compare_raw_addresses([raw_address_one])

    #compare two addresses
    delta_dict_raw_addresses_one_two = address_comparer.compare_raw_addresses([raw_address_one, raw_address_three])

    #compare three addresses
    delta_dict_raw_addresses_one_two_three = address_comparer.compare_raw_addresses([raw_address_one, raw_address_two, raw_address_three])


    #delta_dict_from_dict = address_comparer.delta_dict_from_dict({'deeparse_one' :list_of_tuples_address_one,
    #                            'deeparse_two' :list_of_tuples_address_two})

    
