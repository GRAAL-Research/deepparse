from typing import List, Union

class FormatedComparedAddress:

    def __init__(self, raw_address, parsed_tuples, list_of_bool):
        """
        Address parser used to parse the addresses
        """
        self.raw_address = raw_address
        self.parsed_tuples = parsed_tuples
        self.list_of_bool = list_of_bool
        self.equivalent = self._equivalent()


    def __str__(self) -> str:
        return f"Compared addresses"

    __repr__ = __str__  # to call __str__ when list of address

    def _equivalent(self) ->bool:
        return all([bool_address[1] for bool_address in self.list_of_bool])