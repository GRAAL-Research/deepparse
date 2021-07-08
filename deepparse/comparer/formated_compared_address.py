from typing import List, Union, Tuple

class FormatedComparedAddress:

    def __init__(self, raw_address: Union[List[str], str], parsed_tuples : List[List[Tuple]], list_of_bool: List[Tuple[str, bool]]) -> None:
        """
        Address parser used to parse the addresses
        """
        self.raw_address = raw_address
        self.parsed_tuples = parsed_tuples
        self.list_of_bool = list_of_bool
        self.equivalent = self._equivalent()


    def __str__(self) -> str:
        return "Compared addresses"

    __repr__ = __str__  # to call __str__ when list of address

    def _equivalent(self) ->bool:
        return all([bool_address[1] for bool_address in self.list_of_bool])