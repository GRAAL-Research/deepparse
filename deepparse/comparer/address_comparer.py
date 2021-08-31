from typing import List, Tuple, Union, Dict

from .formated_compared_address import FormatedComparedAddress
from ..parser import AddressParser


# ça responsabilité est de comparer des adresses avec notre parsing d'adresse
# AdressComparer().compare([(addresse_1, [parsing]), (addresse_2, [parsing]), ..., (addresse_n, [parsing])])
# J'aimerai avoir à la sortie une liste de N objets d'adresse comparé
# L'objet outputer est comme un conteneur qui contient l'adresse original, la différence en liste de bool de la longueur
# du nombre de tag et une méthode __str__ pour afficher le output différent.
class AdressComparer:
    """
        Address comparer to compare addresses with each other and retrieves the differences between them. The addresses
        are parsed using an address parser based on one of the seq2seq pre-trained
        networks either with fastText or BPEmb

        The address comparer is able to compare already parsed addresses,
        the address parser first recompose the raw address then suggests its own tags,
        then it makes a comparison with the tags of the source parsing and the
        newly parsed address

        The address comparer is also able to compare raw addresses.
        First it parse the addresses and then bring out the differences
        among the parsed addresses

    Args:
        parser (AddressParser): the AddressParser used to parse the addresses
        colorblind (bool): if True, the differences among the parsed addresses will
                            be shown in a colorblind friendly mode, default value is False

    Examples:
    """

    def __init__(self, parser: AddressParser, colorblind:bool = None) -> None:
        self.parser = parser
        self.__colorblind = False if colorblind is None else colorblind

    def __str__(self) -> str:
        return f"Compare addresses with {self.parser.model_type.capitalize()}AddressParser"

    __repr__ = __str__  # to call __str__ when list of address

    def _compare(self,
                formated_addresses_to_compare:
                Union[Dict, List[Dict]]) -> Union[List[FormatedComparedAddress],FormatedComparedAddress]:
        """Fonction to create a list of FormatedComparedAddress object with the addresses
        to be compared, it is the same process either it is tags comparison or raw addresses
        comparison

        Args:
            addresses_to_compare (Union[Dict, List[Dict]]): formated informations
                for the addresses to be compared. it is possible to have only one comparison
                to make and it will be in the Dict format on it is possible to make multiples
                comparisons and it will take the List[Dict] format.

        Return:
            Either a :class:`~FormattedComparedAddress` or a list of
            :class:`~FormattedComparedAddress` when given more than one comparison to make.

        Examples:
        """
        if isinstance(formated_addresses_to_compare, dict):
            formated_addresses_to_compare = [formated_addresses_to_compare]

        list_of_formated_compared_address = []
        for formated_address_to_compare in formated_addresses_to_compare:
            list_of_formated_compared_address.append(FormatedComparedAddress(formated_address_to_compare,
                                                                            self.__colorblind))
        return list_of_formated_compared_address if len(list_of_formated_compared_address) > 1 \
                else list_of_formated_compared_address[0]

    def compare_tags(self,
                    addresses_to_compare: Union[List[tuple], List[List[tuple]]]
                    )-> Union[List[FormatedComparedAddress],FormatedComparedAddress]:
        """
        Compare tags of a source parsing with the parsing from addressParser. First it recontructs the
        raw address from the parsing, then addressParser generates tags and then compare the tags

        Args:
            addresses_to_compare (Union[List[List[tuple]], List[tuple]]): Takes a list of tuple that contains
            the tags for the address components from the source. Can compare multiples parsings if passed as a
            list of list of tuples

        Return:
            Either a :class:`~FormattedComparedAddress` or a list of
            :class:`~FormattedComparedAddress` when given more than one comparison to make.
        """

        if isinstance(addresses_to_compare[0], tuple):
            addresses_to_compare = [addresses_to_compare]

        rebuilt_raw_addresses = self._get_raw_addresses(addresses_to_compare)

        deepparsed_addresses = self.parser(rebuilt_raw_addresses, with_prob=True)

        if not isinstance(deepparsed_addresses, list):
            deepparsed_addresses = [deepparsed_addresses]

        list_of_deepparse_tuple = []
        List_of_list_of_prob = []
        for parsed_address in deepparsed_addresses:
            dict_of_attr = parsed_address.to_dict()
            list_of_deepparse_tuple.append([(value, key) for key, value in dict_of_attr.items()])
            List_of_list_of_prob.append(parsed_address.address_parsed_components)


        list_of_informations_dict = self._format_info_for_tags_comparison(
        rebuilt_raw_addresses,
        addresses_to_compare,
        list_of_deepparse_tuple,
        List_of_list_of_prob,
        deepparsed_addresses)

        return self._compare(list_of_informations_dict)

    def _format_info_for_tags_comparison(self,
                                            rebuilt_raw_addresses,
                                            addresses_to_compare,
                                            list_of_deepparse_tuple,
                                            List_of_list_of_prob,
                                            deepparsed_addresses) -> Union[Dict, List[Dict]]:

        list_of_informations_dict = []

        for raw_address, address_to_compare, deepparsed_tuple, list_of_prob, deepparsed_address in \
            zip(rebuilt_raw_addresses,
                addresses_to_compare,
                list_of_deepparse_tuple,
                List_of_list_of_prob,
                deepparsed_addresses):
            address_dict = {"raw_addresses" : [raw_address],
                            "address_one":{"tags":address_to_compare,
                                                "origin":"source"},
                            "address_two":{"tags": deepparsed_tuple,
                                            "origin":"deepparse using " + self.parser.model_type.capitalize(),
                                            "repr": repr(deepparsed_address),
                                            "probs": list_of_prob},
                            "type_of_comparison" :"tag"}

            list_of_informations_dict.append(address_dict)

        return list_of_informations_dict

    def compare_raw(self, list_of_addresses_to_compare: Union[Tuple[str], List[Tuple[str]]]) -> List[
        FormatedComparedAddress]:
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

        list_of_deeparsed_addresses = []
        for addresses_to_compare in list_of_addresses_to_compare:
            if len(addresses_to_compare) != 2:
                raise ValueError("You need to compare two addresses")

            list_of_deeparsed_addresses.append(self.parser(addresses_to_compare, with_prob=True))

        list_of_list_of_deepparse_tuple = []
        list_of_list_of_list_of_prob = []
        for deepparsed_addresses in list_of_deeparsed_addresses:
            list_of_deepparse_tuple = []
            List_of_list_of_prob = []
            for parsed_address in deepparsed_addresses:
                dict_of_attr = parsed_address.to_dict()
                list_of_deepparse_tuple.append([(value, key) for key, value in dict_of_attr.items()])
                List_of_list_of_prob.append(parsed_address.address_parsed_components)

            list_of_list_of_deepparse_tuple.append(list_of_deepparse_tuple)
            list_of_list_of_list_of_prob.append(List_of_list_of_prob)


        list_of_informations_dict = self._format_info_for_raw_comparison(list_of_addresses_to_compare,
        list_of_list_of_deepparse_tuple,
        list_of_deeparsed_addresses,
        list_of_list_of_list_of_prob
        )
        return self._compare(list_of_informations_dict)


    def _format_info_for_raw_comparison(self,
                                        list_of_addresses_to_compare,
                                        list_of_list_of_deepparse_tuple,
                                        list_of_deeparsed_addresses,
                                        list_of_list_of_list_of_prob) -> Union[Dict, List[Dict]]:

        list_of_informations_dict = []

        for raw_addresses_to_compare, list_of_deepparsed_tuple, deepparsed_address, list_of_list_of_prob in \
            zip(list_of_addresses_to_compare,
                list_of_list_of_deepparse_tuple,
                list_of_deeparsed_addresses,
                list_of_list_of_list_of_prob):

            address_dict = {"raw_addresses" : raw_addresses_to_compare,
                            "address_one":{"tags": list_of_deepparsed_tuple[0],
                                            "origin":"deepparse using " + self.parser.model_type.capitalize(),
                                            "repr": repr(deepparsed_address[0]),
                                            "probs": list_of_list_of_prob[0]},
                            "address_two":{"tags": list_of_deepparsed_tuple[1],
                                            "origin":"deepparse using " + self.parser.model_type.capitalize(),
                                            "repr": repr(deepparsed_address[1]),
                                            "probs": list_of_list_of_prob[1]},
                            "type_of_comparison": "raw"}

            list_of_informations_dict.append(address_dict)

        return list_of_informations_dict



    def _get_raw_addresses(self, parsed_addresses):

        if not isinstance(parsed_addresses[0], list):
            parsed_addresses = [parsed_addresses]

        return [" ".join([element[0] for element in address]) for address in parsed_addresses]


if __name__ == '__main__':
    list_of_tuples_address_one = [("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                    ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_address_two = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                    ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    raw_address_one = "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"
    raw_address_two = "350 rue des Lilas Ouest Québec Québec G1L 1B6"
    raw_address_three = "450 rue des Lilas Ouest Quebec city Quebec G1L 1B6"

    # test = list(d.compare(raw_address_one, raw_address_three))
    # pprint(test)

    address_parser = AddressParser(model_type="bpemb", device=1)
    address_comparer = AdressComparer(address_parser)

    # Compare with source tags with deepparse tags
    #delta_dict_deeparse_one = address_comparer.compare_tags(list_of_tuples_address_one)

    #delta_dict_deeparse_one_two = address_comparer.compare_tags(
    #    [list_of_tuples_address_one, list_of_tuples_address_two])

    #delta_dict_deeparse_one_two[0].comparison_report()
    #delta_dict_deeparse_one_two[1].comparison_report()

    #compare two equivalent addresses
    delta_dict_raw_addresses_one_two = address_comparer.compare_raw((raw_address_one, raw_address_three))
    delta_dict_raw_addresses_one_two.comparison_report()


    #compare two not equivalent addresses
    delta_dict_raw_addresses_one_three = address_comparer.compare_raw((raw_address_one, raw_address_three))
    delta_dict_raw_addresses_one_three.comparison_report()




    address_comparer_cb = AdressComparer(address_parser, colorblind=True)
    # Compare with source tags with deepparse tags
    delta_dict_deeparse_one = address_comparer_cb.compare_tags(list_of_tuples_address_one)

    delta_dict_deeparse_one_two = address_comparer_cb.compare_tags(
        [list_of_tuples_address_one, list_of_tuples_address_two])

    delta_dict_deeparse_one_two[0].comparison_report()
    delta_dict_deeparse_one_two[1].comparison_report()

    #compare two equivalent addresses
    delta_dict_raw_addresses_one_two = address_comparer_cb.compare_raw((raw_address_one, raw_address_two))
    delta_dict_raw_addresses_one_two.comparison_report()

    #compare two not equivalent addresses
    delta_dict_raw_addresses_one_three = address_comparer_cb.compare_raw((raw_address_one, raw_address_three))
    delta_dict_raw_addresses_one_three.comparison_report()
