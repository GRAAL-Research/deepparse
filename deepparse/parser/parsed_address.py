from typing import Dict, List, Tuple


class FormattedParsedAddress:
    """
    A parsed address as commonly known returned by an address parser.

    Args:
        address_components (List): A list of the address components to tag the address such as StreetName, StreetNumber,
            etc.
        address (dict): A dictionary where the key is an address, and the value is a list of tuples where
            the first elements are address components, and the second elements are the parsed address
            value. Also, the second tuple's address value can either be the tag of the components
            (e.g. StreetName) or a tuple (``x``, ``y``) where ``x`` is the tag and ``y`` is the
            probability (e.g. 0.9981) of the model prediction.

    Note:
        Since an address component can be composed of multiple elements (e.g. Wolfe street), when the probability
        values are asked of the address parser, the address components don't keep it. It's only available through the
        ``address_parsed_components`` attribute.

    Attributes:
        raw_address: The raw address (not parsed).
        address_parsed_components: The parsed address in a list of tuples where the first elements
            are the address components and the second elements are the tags.
        <Address tag>: All the possible address tag element of the model such as StreetName, StreetNumber, etc.

    Example:

        .. code-block:: python

                address_parser = AddressParser()
                parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
                print(parse_address.StreetNumber) # 350
                print(parse_address.PostalCode) # G1L 1B6
    """

    def __init__(self, address_components: List, address: Dict):
        for key in address_components:
            setattr(self, key, None)

        self.raw_address = list(address.keys())[0]
        self.address_parsed_components = address[self.raw_address]

        self._resolve_tagged_affectation(self.address_parsed_components)

    def __str__(self) -> str:
        return self.raw_address

    def _resolve_tagged_affectation(self, tagged_address: List[Tuple]) -> None:
        """
        Private method to resolve the parsing of the tagged address.
        Args:
             tagged_address: The tagged address where the keys are the address component and the values are the
                associated tag.
        """
        for address_component, tag in tagged_address:
            if isinstance(tag, tuple):  # when tag is also the tag and the probability of the tag
                tag = tag[0]

            if getattr(self, tag) is None:
                # empty address components
                setattr(self, tag, address_component)
            else:
                # we merge the previous components with the new element
                setattr(self, tag, " ".join(
                    [getattr(self, tag), address_component]))

    def _get_attr_repr(self, name):
        value = getattr(self, name)
        if value is not None:
            return name + "=" + repr(getattr(self, name))
        return ""

    def __repr__(self):
        values = [self._get_attr_repr(name) for name in self.__dict__ if
                  name != "raw_address" and name != "address_parsed_components"]
        joined_values = ", ".join(v for v in values if v != "")
        return self.__class__.__name__ + "<" + joined_values + ">"
