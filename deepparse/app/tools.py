from typing import Dict, List, Union

from decouple import config
from fastapi import HTTPException

from deepparse.app.address import Address
from deepparse.download_tools import MODEL_MAPPING_CHOICES
from deepparse.parser import AddressParser

address_parser_mapping: Dict[str, AddressParser] = {}

# Upper bound on the number of addresses accepted per request, to avoid resource exhaustion (DoS) from an
# arbitrarily large payload. Overridable via the MAX_ADDRESSES_PER_REQUEST environment variable.
MAX_ADDRESSES_PER_REQUEST = config("MAX_ADDRESSES_PER_REQUEST", 1024, cast=int)


def format_parsed_addresses(
    parsing_model: str, addresses: List[Address], model_mapping=None
) -> Dict[str, Union[str, List[Dict[str, Dict[str, str]]]]]:
    """
    Format parsed addresses.

    Args:
    - **parsing_model** (str): The parsing model to use for address parsing.
    - **addresses** (List[Address]): List of addresses to parse.

    Returns:
    - **JSONResponse**: JSON response containing the parsed addresses, along with the model type and version.
    """
    # These are raised as HTTPException (not ValueError) because this function runs as a FastAPI dependency:
    # a ValueError would surface as a 500 error instead of the documented 422.
    if not addresses:
        raise HTTPException(status_code=422, detail="Addresses parameter must not be empty")
    if len(addresses) > MAX_ADDRESSES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many addresses in a single request (max {MAX_ADDRESSES_PER_REQUEST}).",
        )
    if parsing_model not in MODEL_MAPPING_CHOICES:
        raise HTTPException(
            status_code=422, detail=f"Parsing model not implemented, available choices: {list(MODEL_MAPPING_CHOICES)}"
        )

    if model_mapping is None:
        model_mapping = address_parser_mapping

    parsed_addresses = model_mapping[parsing_model]([address.raw for address in addresses])

    if not isinstance(parsed_addresses, list):
        parsed_addresses = [parsed_addresses]

    response_payload = {
        "model_type": model_mapping[parsing_model].model_type,
        # A list (not a dict keyed on the raw address) so that duplicate input addresses are preserved and the
        # output order matches the input order.
        "parsed_addresses": [
            {raw_address.raw: parsed_address.to_dict()}
            for parsed_address, raw_address in zip(parsed_addresses, addresses)
        ],
        "version": model_mapping[parsing_model].version,
    }

    return response_payload
