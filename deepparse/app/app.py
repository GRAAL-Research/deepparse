"""REST API."""
from typing import List, Dict, Union
from contextlib import asynccontextmanager

from pydantic import BaseModel
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse

from deepparse.cli import download_models_test
from deepparse.tools import CACHE_PATH, MODEL_CHOICES
from deepparse.parser import AddressParser

address_parser_mapping: Dict[str, AddressParser] = {}


@asynccontextmanager
async def lifespan(application: FastAPI):  # pylint: disable=unused-argument
    # Load the models
    download_models_test(CACHE_PATH)
    for model in MODEL_CHOICES:
        if model not in ["fasttext", "fasttext-attention"]:
            attention = False
            if "-attention" in model:
                attention = True
            address_parser_mapping[model] = AddressParser(
                model_type=model.replace("-", "_"),
                offline=True,
                attention_mechanism=attention,
                device="cpu",
            )
    yield
    # Clean up the address parsers and release the resources
    address_parser_mapping.clear()


app = FastAPI(lifespan=lifespan)


class Address(BaseModel):
    raw: str


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
        parsing_model in MODEL_CHOICES
    ), f"Parsing model not implemented, available choices: {MODEL_CHOICES}"

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


@app.post("/parse/{parsing_model}")
def parse(parsing_model: str, addresses: List[Address], resp=Depends(format_parsed_addresses)):
    """
    Parse addresses using the specified parsing model.

    Args:
    - **parsing_model** (str, path parameter): The parsing model to use for address parsing.
      Available choices: fasttext, fasttext-attention, fasttext-light, bpemb, bpemb-attention
    - **addresses** (List[Address], request body): List of addresses to parse.

    Returns:
    - **JSONResponse**: JSON response containing the parsed addresses, along with the model type and version.

    Raises:
    - **AssertionError**: If the addresses parameter is empty or if the specified parsing model is not implemented.

    Examples:
        Python Requests:

        import requests

        url = 'http://localhost:8000/parse/bpemb'
        addresses = [
            {"raw": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
            {"raw": "2325 Rue de l'Université, Québec, QC G1V 0A6"}
        ]

        response = requests.post(url, json=addresses)
        parsed_addresses = response.json()
        print(parsed_addresses)
    """
    assert addresses, "Addresses parameter must not be empty"
    assert (
        parsing_model in MODEL_CHOICES
    ), f"Parsing model not implemented, available choices: {MODEL_CHOICES}"
    return JSONResponse(content=resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
