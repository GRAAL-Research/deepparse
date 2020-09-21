from typing import Dict


class ParsedAddress:
    """
    A parsed address as commonly known returned by an address parser.

    Note:
        Since an address component can be composed of multiple elements (e.g. Wolfe street), when the probability
        values are asked to the address parser, the address components don't keep it. It's only available through the
        ``address_parsed_dict`` attribute.

    Attributes:
        raw_address: The raw address (not parsed).
        address_parsed_dict: The parsed address in a dictionary where the keys are the address parsed
            and the values is a dictionary of the parsed components.
        street_number: The street number.
        unit: The street unit component.
        street_name: The street name.
        orientation: The street orientation.
        municipality: The city.
        province: The province (sometime known as district or local region).
        postal_code: The street postal code.
        general_delivery: Additional delivery information.

    Example:

        .. code-block:: python

                address_parser = AddressParser()
                parse_address = address_parser('350 rue des Lilas Ouest Quebec city Quebec G1L 1B6')
                print(parse_address.street_number) #350
                print(parse_address.postal_code) # G1L 1B6
    """

    def __init__(self, address: Dict):
        """
        Args:
            address: A dictionary where the key is an address, and the value is another dictionary where the keys are
                address components, and the values are the parsed address value. Also, the second dictionary's
                address value can either be the tag of the components (e.g. StreetName) or a tuple (``x``, ``y``)
                where ``x`` is the tag and ``y`` is the probability (e.g. 0.9981) of the model prediction.
        """
        self.raw_address = list(address.keys())[0]
        self.address_parsed_dict = address[self.raw_address]

        self.street_number = None
        self.unit = None
        self.street_name = None
        self.orientation = None
        self.municipality = None
        self.province = None
        self.postal_code = None
        self.general_delivery = None

        self._resolve_tagged_affectation(self.address_parsed_dict)

    def __str__(self) -> str:
        return self.raw_address

    def _resolve_tagged_affectation(self, tagged_address: Dict) -> None:
        """
        Private method to resolve the parsing of the tagged address.
        :param tagged_address: The tagged address where the keys are the address component and the values are the
        associated tag.
        """
        for address_component, tag in tagged_address.items():
            if isinstance(tag, tuple):  # when tag is also the tag and the probability of the tag
                tag = tag[0]
            if tag == "StreetNumber":
                self.street_number = address_component if self.street_number is None else " ".join(
                    [self.street_number, address_component])
            elif tag == "Unit":
                self.unit = address_component if self.unit is None else " ".join(
                    [self.unit, address_component])
            elif tag == "StreetName":
                self.street_name = address_component if self.street_name is None else " ".join(
                    [self.street_name, address_component])
            elif tag == 'Orientation':
                self.orientation = address_component if self.orientation is None else " ".join(
                    [self.orientation, address_component])
            elif tag == 'Municipality':
                self.municipality = address_component if self.municipality is None else " ".join(
                    [self.municipality, address_component])
            elif tag == 'Province':
                self.province = address_component if self.province is None else " ".join(
                    [self.province, address_component])
            elif tag == 'PostalCode':
                self.postal_code = address_component if self.postal_code is None else " ".join(
                    [self.postal_code, address_component])
            elif tag == 'GeneralDelivery':
                self.general_delivery = address_component if self.general_delivery is None else " ".join(
                    [self.general_delivery, address_component])

    __repr__ = __str__  # to call __str__ when list of address
