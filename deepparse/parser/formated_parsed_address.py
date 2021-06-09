from typing import Dict, List, Tuple, Union

FIELDS = [
    "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode", "GeneralDelivery"
]


class FormattedParsedAddress:
    """
    A parsed address as commonly known returned by an address parser.

    Args:
        address (dict): A dictionary where the key is an address, and the value is a list of tuples where
            the first elements are address components, and the second elements are the parsed address
            value. Also, the second tuple's address value can either be the tag of the components
            (e.g. StreetName) or a tuple (``x``, ``y``) where ``x`` is the tag and ``y`` is the
            probability (e.g. 0.9981) of the model prediction.

    Attributes:
        raw_address: The raw address (not parsed).
        address_parsed_components: The parsed address in a list of tuples where the first elements
            are the address components and the second elements are the tags.
        <Address tag>: All the possible address tag element of the model. For example, ``StreetName`` or
            ``StreetNumber``.

    Example:

        .. code-block:: python

            address_parser = AddressParser()
            parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
            print(parse_address.StreetNumber) # 350
            print(parse_address.PostalCode) # G1L 1B6

    Note:
        Since an address component can be composed of multiple elements (e.g. Wolfe street), when the probability
        values are asked of the address parser, the address components don't keep it. It's only available through the
        ``address_parsed_components`` attribute.
    """

    def __init__(self, address: Dict):
        for key in FIELDS:
            setattr(self, key, None)

        self.raw_address = list(address.keys())[0]
        self.address_parsed_components = address[self.raw_address]

        self._resolve_tagged_affectation(self.address_parsed_components)

    def __str__(self) -> str:
        return self.raw_address

    def to_dict(self, fields: Union[List, None] = None) -> dict:
        """
        Method to convert a parsed address into a dictionary where the keys are the address components and the values
        are the value of those components. For example, the parsed address ``<StreetNumber> 305 <StreetName>
        rue des Lilas`` will be converted into the following dictionary:
        ``{'StreetNumber':'305', 'StreetName': 'rue des Lilas'}``.

        Args:
            fields (Union[list, None]): Optional argument to define the fields to extract from the address and the
                order of it. If None, will used the default order and value `'StreetNumber, Unit, StreetName,
                Orientation, Municipality, Province, PostalCode, GeneralDelivery'`.

        Return:
            A dictionary where the keys are the selected (or default) fields and the values are the corresponding value
            of the address components.
        """
        if fields is None:
            fields = FIELDS
        return {field: getattr(self, field) for field in fields}

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
                setattr(self, tag, " ".join([getattr(self, tag), address_component]))

    def _get_attr_repr(self, name):
        value = getattr(self, name)
        if value is not None:
            return name + "=" + repr(getattr(self, name))
        return ""

    def __repr__(self):
        values = [
            self._get_attr_repr(name) for name in self.__dict__
            if name not in ("raw_address", "address_parsed_components")
        ]
        joined_values = ", ".join(v for v in values if v != "")
        return self.__class__.__name__ + "<" + joined_values + ">"
