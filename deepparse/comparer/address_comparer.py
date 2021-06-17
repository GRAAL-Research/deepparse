from deepparse.parser import AddressParser
from typing import List


class AdressComparer:
    """
        Compares two addresses and return the the differences between them
    """
    
    def __init__(self, parser: AddressParser) -> None:
        """
        base address to make the comparisons
        """

        self.parser = parser

    
    def compare(self, address_to_compare: List[tuple]) -> bool:
        """
        305 rue des Lilas O, app 2 
        StreetNumber, StreetName ...
        [(305, StreetNumber), (rue, StreetName), ...]
        305 rue ...
        #{305: StreetNumber} risque de colission keys

        StreetName, StreetNumber

        """
        
        rebuilt_raw_address = " ".join([element[0] for element in address_to_compare])
        deepparsed_address = self.parser(rebuilt_raw_address)

        list_of_attr = deepparsed_address.to_dict()
        for key, value in list_of_attr:
            print(key)
            print(value)
            #getattr(parsed_address, element)
            #comparaison ...

        return True

        
        #return (self._compare_streetNumber(compared_address.StreetNumber) and
        #    self._compare_StreetName(compared_address.StreetName) and
        #    self._compare_Unit(compared_address.Unit) and
        #    self._compare_Municipality(compared_address.Municipality) and
        #    self._compare_Province(compared_address.Province) and
        #    self._compare_PostalCode(compared_address.PostalCode) and
        #    self._compare_Orientation(compared_address.Orientation) and
        #    self._compare_GeneralDelivery(compared_address.GeneralDelivery))

    #def _compare_streetNumber(self, compared_streetNumber: str) -> bool:
    #    return self.StreetNumber == compared_streetNumber

    #def _compare_StreetName(self, compared_StreetName: str) -> bool:
    #    return self.StreetName == compared_StreetName

    #def _compare_Unit(self, compared_Unit: str) -> bool:
    #    return self.Unit == compared_Unit

    #def _compare_Municipality(self, compared_Municipality: str) -> bool:
    #    return self.Municipality == compared_Municipality

    #def _compare_Province(self, compared_Province: str) -> bool:
    #    return self.Province == compared_Province

    #def _compare_PostalCode(self, compared_PostalCode: str) -> bool:
    #    return self.PostalCode == compared_PostalCode

    #def _compare_Orientation(self, compared_Orientation: str) -> bool:
    #    return self.Orientation == compared_Orientation

    #def _compare_GeneralDelivery(self, compared_GeneralDelivery: str) -> bool:
    #    return self.GeneralDelivery == compared_GeneralDelivery

    
    #def delta_dict(self, compared_address: AddressParser) -> dict:
    #    delta_dict = {}
#
    #    if not self._compare_streetNumber(compared_address.StreetNumber):
    #        delta_dict['StreetNumber'] = {'base' : self.StreetNumber, 'compared' : compared_address.StreetNumber}
#
    #    if not self._compare_StreetName(compared_address.StreetName):
    #        delta_dict['StreetName'] = {'base' : self.StreetName, 'compared' : compared_address.StreetName}
#
    #    if not self._compare_Unit(compared_address.Unit):
    #        delta_dict['Unit'] = {'base' : self.Unit, 'compared' : compared_address.Unit}
#
    #    if not self._compare_Municipality(compared_address.Municipality):
    #        delta_dict['Municipality'] = {'base' : self.Municipality, 'compared' : compared_address.Municipality}
#
    #    if not self._compare_Province(compared_address.Province):
    #        delta_dict['Province'] = {'base' : self.Province, 'compared' : compared_address.Province}
#
    #    if not self._compare_PostalCode(compared_address.PostalCode):
    #        delta_dict['PostalCode'] = {'base' : self.PostalCode, 'compared' : compared_address.PostalCode}
#
    #    if not self._compare_Orientation(compared_address.Orientation):
    #        delta_dict['Orientation'] = {'base' : self.Orientation, 'compared' : compared_address.Orientation}
#
    #    if not self._compare_GeneralDelivery(compared_address.GeneralDelivery):
    #        delta_dict['GeneralDelivery'] = {'base' : self.GeneralDelivery, 'compared' : compared_address.GeneralDelivery}
#
    #    return delta_dict


if __name__ == '__main__':

    address_parser = AddressParser(model_type="bpemb", device=0)

    # you can parse one address
    #parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

    #parsed_address_same = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")
    #parsed_address_diff = address_parser("450 rue des Lilas Ouest Québec Québec G1L 1B6")


    address_comparer = AdressComparer(address_parser)
    address_comparer.compare([("305", "StreetNumber"), ("rue des Lilas", "StreetName"), ("Ouest", "Orientation"),
                                ("Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")])

    #test == parsed_address_same
    #test == parsed_address_diff


    #parsed_address_diff_streetNumber = address_parser("450 rue des Lilas Ouest Québec Québec G1L 1B6")
    #parsed_address_diff_streetName = address_parser("350 Boulevard des Lilas Ouest Québec Québec G1L 1B6")
    #parsed_address_diff_Unit = address_parser("350 rue des Lilas Ouest app 105 Québec Québec G1L 1B6")
    #parsed_address_diff_Municipality = address_parser("350 rue des Lilas Ouest Ste-Foy Québec G1L 1B6")
    #parsed_address_diff_Province = address_parser("350 rue des Lilas Ouest Québec Ontario G1L 1B6")
    #parsed_address_diff_PostalCode = address_parser("350 rue des Lilas Ouest Québec Québec G1P 1B6")
    #parsed_address_diff_Orientation = address_parser("350 rue des Lilas Est Québec Québec G1L 1B6")

    #dict_vide = test.delta_dict(parsed_address_same)
    #dict_numberStreet_diff = test.delta_dict(parsed_address_diff)

    #dict_parsed_address_diff_streetNumber = test.delta_dict(parsed_address_diff_streetNumber) 
    #dict_parsed_address_diff_streetName = test.delta_dict(parsed_address_diff_streetName) 
    #dict_parsed_address_diff_Unit = test.delta_dict(parsed_address_diff_Unit) 
    #dict_parsed_address_diff_Municipality = test.delta_dict(parsed_address_diff_Municipality) 
    #dict_parsed_address_diff_Province = test.delta_dict(parsed_address_diff_Province) 
    #dict_parsed_address_diff_PostalCode = test.delta_dict(parsed_address_diff_PostalCode) 
    #dict_parsed_address_diff_Orientation = test.delta_dict(parsed_address_diff_Orientation) 