from typing import List, Dict, Union

from deepparse.app.address import Address
from deepparse.download_tools import MODEL_MAPPING_CHOICES
from deepparse.parser import AddressParser

address_parser_mapping: Dict[str, AddressParser] = {}


def format_parsed_addresses(
    parsing_model: str, addresses: List[Address], model_mapping=None
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Format parsed addresses.

    Args:
    - **parsing_model** (str): The parsing model to use for address parsing.
    - **addresses** (List[Address]): List of addresses to parse.

    Returns:
    - **JSONResponse**: JSON response containing the parsed addresses, along with the model type and version.
    """
    assert addresses, "Addresses parameter must not be empty"
    assert (
        parsing_model in MODEL_MAPPING_CHOICES
    ), f"Parsing model not implemented, available choices: {MODEL_MAPPING_CHOICES}"

    if model_mapping is None:
        model_mapping = address_parser_mapping

    parsed_addresses = model_mapping[parsing_model]([address.raw for address in addresses])

    if not isinstance(parsed_addresses, list):
        parsed_addresses = [parsed_addresses]

    response_payload = {
        "model_type": model_mapping[parsing_model].model_type,
        "parsed_addresses": {
            raw_address.raw: parsed_address.to_dict()
            for parsed_address, raw_address in zip(parsed_addresses, addresses)
        },
        "version": model_mapping[parsing_model].version,
    }

    return response_payload
