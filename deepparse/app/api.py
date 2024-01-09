"""REST API."""
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from enum import Enum


try:
    from pydantic import BaseModel, Field
    from fastapi import FastAPI, Depends
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from deepparse.download_tools import MODEL_MAPPING_CHOICES, download_models
    from deepparse.parser import AddressParser
    from deepparse.app.deepparser_logger import logger
    from deepparse.app.sentry import configure_sentry
    import os


except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Ensure you installed the extra packages using: 'pip install deepparse[app]'") from e


address_parser_mapping: Dict[str, AddressParser] = {}


class ModelName(str, Enum):
    bpemb = "bpemb"
    bpemb_attention = "bpemb-attention"
    fasttext = "fasttext"
    fasttext_attention = "fasttext-attention"
    fasttext_light = "fasttext-light"


class AddressInput(BaseModel):
    address: str = Field(examples=["350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"])


def initialize_address_parser_mapping() -> Dict[str, AddressParser]:
    _address_parser_mapping: Dict[str, AddressParser] = {}
    logger.debug("Downloading models")
    download_models()
    for model in MODEL_MAPPING_CHOICES:
        logger.debug("initializing %s", model)
        attention = False
        if "-attention" in model:
            attention = True
        _address_parser_mapping[model] = AddressParser(
            model_type=model,
            offline=True,
            attention_mechanism=attention,
            device="cpu",
        )

    return _address_parser_mapping


class AddressParserService:
    def __init__(self, address_parser_models: Dict[str, AddressParser]) -> None:
        self.address_parser_models = address_parser_models

    def __call__(self, model: str, addresses: List[str]) -> Dict[str, Any]:
        parsed_addresses = self.address_parser_models[model](addresses)

        if not isinstance(parsed_addresses, list):
            parsed_addresses = [parsed_addresses]

        response_payload = {
            "model_type": self.address_parser_models[model].model_type,
            "parsed_addresses": {
                raw_address: parsed_address.to_dict()
                for parsed_address, raw_address in zip(parsed_addresses, addresses)
            },
            "version": self.address_parser_models[model].version,
        }

        return response_payload


@asynccontextmanager
async def lifespan(application: FastAPI):  # pylint: disable=unused-argument
    # Load the models
    address_parser_mapping.update(initialize_address_parser_mapping())
    address_parser_service = AddressParserService(address_parser_mapping)
    yield
    # Clean up the address parsers and release the resources
    address_parser_service.address_parser_models.clear()


def get_address_parser_service() -> AddressParserService:
    return AddressParserService(address_parser_mapping)


configure_sentry()

api = FastAPI(lifespan=lifespan)


@api.post("/parse/{parsing_model}")
def parse(
    parsing_model: ModelName,
    addresses: List[AddressInput],
    address_parser_service: AddressParserService = Depends(get_address_parser_service),
):
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

        url = 'http://localhost:8081/parse/bpemb'
        addresses = [
            {"address": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
            {"address": "2325 Rue de l'Université, Québec, QC G1V 0A6"}
        ]

        response = requests.post(url, json=addresses)
        parsed_addresses = response.json()
        print(parsed_addresses)
    """
    assert addresses, "Addresses parameter must not be empty"
    assert (
        parsing_model in MODEL_MAPPING_CHOICES
    ), f"Parsing model not implemented, available choices: {MODEL_MAPPING_CHOICES}"

    raw_addresses = [address_input.address for address_input in addresses]

    response_payload = address_parser_service(parsing_model, raw_addresses)

    return JSONResponse(content=response_payload)


html_build_path = "docs/_build/html"
if os.path.exists(html_build_path):
    api.mount("/", StaticFiles(directory=html_build_path, html=True), name="static")
else:
    logger.warning("Unable to mount static files, probably because the docs are not built yet: %s", html_build_path)

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8081)
