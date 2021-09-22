from dataclasses import dataclass
from typing import List, Tuple, Union, Dict

from .formatted_compared_addresses_raw import FormattedComparedAddressesRaw
from .formatted_compared_addresses_tags import FormattedComparedAddressesTags
from ..parser import AddressParser
from ..parser.formated_parsed_address import FormattedParsedAddress


# ça responsabilité est de comparer des adresses avec notre parsing d'adresse
# AdressComparer().compare([(addresse_1, [parsing]), (addresse_2, [parsing]), ..., (addresse_n, [parsing])])
# J'aimerai avoir à la sortie une liste de N objets d'adresse comparé
# L'objet outputer est comme un conteneur qui contient l'adresse original, la différence en liste de bool de la longueur
# du nombre de tag et une méthode __str__ pour afficher le output différent.
@dataclass
class AddressesComparer:
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

    Examples:
    list_of_tuples = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                  ("Ouest Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_with_probs =  [
            ('350', ('StreetNumber', 1.0)),
            ('rue', ('StreetName', 0.9987)),
            ('des', ('StreetName', 0.9993)),
            ('Lilas', ('StreetName', 0.8176)),
            ('Ouest', ('Orientation', 0.781)),
            ('Quebec', ('Municipality', 0.9768)),
            ('Quebec', ('Province', 1.0)),
            ('G1L', ('PostalCode', 0.9993)),
            ('1B6', ('PostalCode', 1.0))]

    address_parser = AddressParser(model_type="bpemb", device=1)
    addresses_comparer = AddressesComparer(address_parser)

    delta_dict_deeparse_one_two = addresses_comparer.compare_tags([list_of_tuples,
                                                                   list_of_tuples_with_probs])

    delta_dict_deeparse_one_two[0].comparison_report()
    delta_dict_deeparse_one_two[1].comparison_report()

    raw_address_original = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_identical = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_equivalent = "350  rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_diff_streetNumber = "450 rue des Lilas Ouest Quebec Quebec G1L 1B6"

    raw_addresses_multiples_comparisons = addresses_comparer.compare_raw([(raw_address_original,
                                                                            raw_address_identical)
                                                                            ,(raw_address_original,
                                                                            raw_address_equivalent),
                                                                           (raw_address_original,
                                                                            raw_address_diff_streetNumber)])
    raw_addresses_multiples_comparisons[0].comparison_report()
    raw_addresses_multiples_comparisons[1].comparison_report()
    raw_addresses_multiples_comparisons[2].comparison_report()

    """
    parser: AddressParser

    def __str__(self) -> str:
        return f"Compare addresses with {self.parser.model_type.capitalize()}AddressParser"

    __repr__ = __str__  # to call __str__ when list of address

    def compare_tags(self,
                     addresses_tags_to_compare: Union[List[tuple], List[List[tuple]]],
                     with_probs: Union[None, bool] = None
                     ) -> Union[List[FormattedComparedAddressesTags], FormattedComparedAddressesTags]:

        """ Compare tags of a source parsing with the parsing from AddressParser. First it recontructs the
        raw address from the parsing, then AddressParser generates tags and then compare the two parsing.

        Args:
            addresses_tags_to_compare (Union[List[tuple], List[List[tuple]]]): list of tuple that contains
            the tags for the address components from the source. Can compare multiples parsings if passed as a
            list of list of tuples

        Return:
            Either a :class:`~FormattedComparedAddressesTags` or a list of
            :class:`~FormattedComparedAddressTags` when given more than one comparison to make.
        """

        if isinstance(addresses_tags_to_compare[0], tuple):
            addresses_tags_to_compare = [addresses_tags_to_compare]

        if with_probs is None:
            with_probs = [self._check_if_with_prob(address) for address in addresses_tags_to_compare]
        else:
            with_probs = [with_probs] * len(addresses_tags_to_compare)
        
        raw_addresses = [" ".join([element[0] for element in address]) for address in addresses_tags_to_compare]

        formatted_addresses = [FormattedParsedAddress({raw_address: address_tags}) for raw_address, address_tags \
                               in zip(raw_addresses, addresses_tags_to_compare)]
                               
        deepparsed_formatted_addresses = [self.parser(raw_addess, with_prob=prob) for raw_addess, prob in zip(raw_addresses, with_probs)]

        if isinstance(deepparsed_formatted_addresses, FormattedParsedAddress):
            deepparsed_formatted_addresses = [deepparsed_formatted_addresses]

        comparison_tuples = list(zip(formatted_addresses, deepparsed_formatted_addresses))

        list_of_comparison_dict = self._format_comparisons_dict(comparison_tuples, ("source",
                                                                                    "deepparse using " + self.parser.model_type.capitalize()),
                                                                                    with_probs)

        formatted_comparisons = [FormattedComparedAddressesTags(**comparison_info) for comparison_info \
                                 in list_of_comparison_dict]
        return formatted_comparisons if len(formatted_comparisons) > 1 else formatted_comparisons[0]

    def compare_raw(self,
                    raw_addresses_to_compare: Union[Tuple[str], List[Tuple[str]]],
                    with_probs: Union[None, bool] = None) -> List[
        FormattedComparedAddressesRaw]:

        """Compare a list of raw addresses together, it starts by parsing the addresses
        with the setted parser and then return the differences between the addresses components
        retrieved with our model.

        Args:
            raw_addresses_to_compare (Union[Tuple[str], List[Tuple[str]]]):  List of string that
                                                                            represent raw addresses.

        Raises:
            ValueError: [description]

        Return:
            Either a :class:`~FormattedComparedAddressesRaw` or a list of
            :class:`~FormattedComparedAddressesRaw` when given more than one comparison to make.
        """
        if isinstance(raw_addresses_to_compare[0], str):
            raw_addresses_to_compare = [raw_addresses_to_compare]

        if with_probs is None:
            with_probs = [True] *len(raw_addresses_to_compare)
        else:
            with_probs = [with_probs] * len(raw_addresses_to_compare)

            
        list_of_deeparsed_addresses = []
        for addresses_to_compare, prob in zip(raw_addresses_to_compare, with_probs):
            if len(addresses_to_compare) != 2:
                raise ValueError("You need to compare two addresses")
            list_of_deeparsed_addresses.append(self.parser(addresses_to_compare, with_prob=prob))

        list_of_comparison_dict = self._format_comparisons_dict(list_of_deeparsed_addresses,
                                                                (
                                                                "deepparse using " + self.parser.model_type.capitalize(),
                                                                "deepparse using " + self.parser.model_type.capitalize()
                                                                ),
                                                                with_probs)

        formatted_comparisons = [FormattedComparedAddressesRaw(**comparison_info) for comparison_info \
                                 in list_of_comparison_dict]

        return formatted_comparisons if len(formatted_comparisons) > 1 else formatted_comparisons[0]

    def _format_comparisons_dict(self, comparison_tuples: Tuple[FormattedParsedAddress, FormattedParsedAddress],
                                 origin_tuple: Tuple[str, str],
                                 with_probs: bool) -> List[Dict]:
        """Return formated dict that contains the two FormatedParsedAddress and the origin name
            tuple and output it in a dict.

        Args:
            comparison_tuples (Tuple[FormattedParsedAddress, FormattedParsedAddress]): A tuple that contains
                                                                                        the two FormattedParsedAddress
            origin_tuple (Tuple[str, str]): A tuple that contains the two origin name for the parsed addreses.

        Returns:
            List[Dict]: list that contrains the formated information to construct the FormatedComparedAddress object.
        """
        list_of_formatted_comparisons_dict = []

        for comparison_tuple, prob in zip(comparison_tuples, with_probs):
            comparison_info = {"first_address": comparison_tuple[0],
                               "second_address": comparison_tuple[1],
                               "origin": origin_tuple,
                               "with_probs": prob}

            list_of_formatted_comparisons_dict.append(comparison_info)

        return list_of_formatted_comparisons_dict
    
    @staticmethod
    def _check_if_with_prob(list_of_tuple):
        return len(list_of_tuple[0][1]) == 2 and isinstance(list_of_tuple[0][1][1], float)

if __name__ == '__main__':
    list_of_tuples = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                  ("Ouest Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]

    list_of_tuples_with_probs =  [
            ('350', ('StreetNumber', 1.0)),
            ('rue', ('StreetName', 0.9987)),
            ('des', ('StreetName', 0.9993)),
            ('Lilas', ('StreetName', 0.8176)),
            ('Ouest', ('Orientation', 0.781)),
            ('Quebec', ('Municipality', 0.9768)),
            ('Quebec', ('Province', 1.0)),
            ('G1L', ('PostalCode', 0.9993)),
            ('1B6', ('PostalCode', 1.0))]

    address_parser = AddressParser(model_type="bpemb", device=1)
    addresses_comparer = AddressesComparer(address_parser)

    delta_dict_deeparse_one_two = addresses_comparer.compare_tags([list_of_tuples,
                                                                   list_of_tuples_with_probs])

    delta_dict_deeparse_one_two[0].comparison_report()
    delta_dict_deeparse_one_two[1].comparison_report()

    raw_address_original = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_identical = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_equivalent = "350  rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_diff_streetNumber = "450 rue des Lilas Ouest Quebec Quebec G1L 1B6"

    raw_addresses_multiples_comparisons = addresses_comparer.compare_raw([(raw_address_original,
                                                                            raw_address_identical)
                                                                            ,(raw_address_original,
                                                                            raw_address_equivalent),
                                                                           (raw_address_original,
                                                                            raw_address_diff_streetNumber)])
    raw_addresses_multiples_comparisons[0].comparison_report()
    raw_addresses_multiples_comparisons[1].comparison_report()
    raw_addresses_multiples_comparisons[2].comparison_report()