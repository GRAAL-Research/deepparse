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

        self._infer_tags_order()

        self._resolve_tagged_affectation(self.address_parsed_components)

    def __str__(self) -> str:
        """
        Return raw address as a string.
        """
        return self.raw_address

    def __repr__(self) -> str:
        values = [
            self._get_attr_repr(name) for name in self.__dict__
            if name not in ("raw_address", "address_parsed_components", "inferred_order")
        ]
        joined_values = ", ".join(v for v in values if v != "")
        return self.__class__.__name__ + "<" + joined_values + ">"

    def __eq__(self, other) -> bool:
        """
        Equal if all address components elements are equals. If attributes are not the same, will return False.
        """
        for field in self.__dict__:
            address_component = getattr(self, field)
            try:
                other_address_component = other.__getattribute__(field)
            except AttributeError:
                # Attribute not the same.
                return False
            if address_component != other_address_component:
                # An element is different.
                return False
        return True

    def format_address(self,
                       fields: Union[List, None] = None,
                       capitalize_fields: Union[List[str], None] = None,
                       upper_case_fields: Union[List[str], None] = None,
                       field_separator: Union[str, None] = None) -> str:
        """
        Method to format the address components in a specific order. We also filter the empty components (None).
        By default, the order is `'StreetNumber, Unit, StreetName, Orientation, Municipality, Province, PostalCode,
        GeneralDelivery'` and we filter the empty components.

        Args:
            fields (Union[list, None]): Optional argument to define the fields to order the address components of
                the address. If None, we will use the inferred order base on the address tags appearance. For example,
                if the parsed address is ``(305, StreetNumber), (rue, StreetName), (des, StreetName),
                (Lilas, StreetName)``, the inferred order will be ``StreetNumber, StreetName``.
            capitalize_fields (Union[list, None]): Optional argument to define the capitalize fields for the formatted
                address. If None, no fields are capitalize.
            upper_case_fields (Union[list, None]): Optional argument to define the upper cased fields for the
                formatted address. If None, no fields are capitalize.
            field_separator (Union[list, None]): Optional argument to define the field separator between address
                components. If None, the default field separator is ``" "``.

        Return:
            A string of the formatted address in the fields order.

        Examples:

            .. code-block:: python

                address_parser = AddressParser()
                parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")

                parse_address.formatted_address(fields_separator=", ")
                # > 350, rue des lilas, ouest, quebec city, quebec, g1l 1b6

                parse_address.formatted_address(fields_separator=", ", capitalize_fields=["StreetName", "Orientation"])
                # > 350, Rue des lilas, Ouest, quebec city, quebec, g1l 1b6

                parse_address.formatted_address(fields_separator=", ", upper_case_fields=["PostalCode""])
                # > 350 rue des lilas ouest quebec city quebec G1L 1B6
        """
        if fields is None:
            fields = self.inferred_order
        self._validate_argument(fields)

        if capitalize_fields is None:
            capitalize_fields = []
        self._validate_argument(capitalize_fields)

        if upper_case_fields is None:
            upper_case_fields = []
        self._validate_argument(upper_case_fields)

        if field_separator is None:
            field_separator = " "

        formatted_parsed_address = ""
        for field in fields:
            address_component = getattr(self, field)
            if address_component is not None:
                # Format address
                address_component = address_component.capitalize() if field in capitalize_fields else address_component
                address_component = address_component.upper() if field in upper_case_fields else address_component

                formatted_parsed_address += address_component + field_separator

        return formatted_parsed_address.strip(field_separator)  # To remove last field separator

    def to_dict(self, fields: Union[List, None] = None) -> dict:
        """
        Method to convert a parsed address into a dictionary where the keys are the address components, and the values
        are the value of those components. For example, the parsed address ``<StreetNumber> 305 <StreetName>
        rue des Lilas`` will be converted into the following dictionary:
        ``{'StreetNumber':'305', 'StreetName': 'rue des Lilas'}``.

        Args:
            fields (Union[list, None]): Optional argument to define the fields to extract from the address and the
                order of it. If None, will use the default order and value `'StreetNumber, Unit, StreetName,
                Orientation, Municipality, Province, PostalCode, GeneralDelivery'`.

        Return:
            A dictionary where the keys are the selected (or default) fields and the values are the corresponding value
            of the address components.
        """
        if fields is None:
            fields = FIELDS
        return {field: getattr(self, field) for field in fields}

    def to_list_of_tuples(self, fields: Union[List, None] = None) -> List[tuple]:
        """
        Method to convert a parsed address into a list of tuples where the first element of the tuples
        is the value of the components, and the second value is the name of the components.

        For example, the parsed address ``<StreetNumber> 305 <StreetName> rue des Lilas`` will be converted into the
        following list of tuples: ``('305', 'StreetNumber'), ('rue des Lilas', 'StreetName')]``.

        Args:
            fields (Union[list, None]): Optional argument to define the fields to extract from the address and its
                order. If None, it will use the default order and value `'StreetNumber, Unit, StreetName,
                Orientation, Municipality, Province, PostalCode, GeneralDelivery'`.

        Return:
            A list of tuples where the first element of the tuples are the value of the address components
            and the second values are the name of the address components.

        """
        dict_of_attr = self.to_dict(fields)
        return [(value, key) for key, value in dict_of_attr.items()]

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

    def _validate_argument(self, arg: List) -> None:
        for arg_element in arg:
            if not hasattr(self, arg_element):
                raise KeyError(arg_element + " not an attribute of the formatted parsed address.")

    def _infer_tags_order(self) -> None:
        """
        Private method to infer the order of the tags base on the address order tag.
        """
        tags = [tag for _, tag in self.address_parsed_components]
        inferred_order = []
        for tag in tags:
            if tag not in inferred_order:
                inferred_order.append(tag)
        self.inferred_order = inferred_order
