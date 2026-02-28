"""REST API."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from deepparse.app.address import Address
from deepparse.app.tools import address_parser_mapping, format_parsed_addresses
from deepparse.download_tools import MODEL_MAPPING_CHOICES, download_models
from deepparse.parser import AddressParser

try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    from deepparse.app.sentry import configure_sentry  # pylint: disable=ungrouped-imports
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Ensure you installed the extra packages using: 'pip install deepparse[app]'") from e

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s; %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.WARNING)

configure_sentry()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:  # pylint: disable=unused-argument
    # Load the models
    logger.debug("Downloading models")
    download_models()
    for model in MODEL_MAPPING_CHOICES:
        if model not in ["fasttext", "fasttext-attention", "fasttext-light"]:  # Skip fasttext models
            logger.debug("initializing %s", model)
            attention = False
            if "-attention" in model:
                attention = True
            address_parser_mapping[model] = AddressParser(
                model_type=model,
                offline=True,
                attention_mechanism=attention,
                device="cpu",
            )
    yield
    # Clean up the address parsers and release the resources
    address_parser_mapping.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/parse/{parsing_model}")
def parse(parsing_model: str, addresses: List[Address], resp: dict = Depends(format_parsed_addresses)) -> JSONResponse:
    """
    Parse addresses using the specified parsing model.

    Args:
    - **parsing_model** (str, path parameter): The parsing model to use for address parsing.
      Available choices: fasttext, fasttext-attention, fasttext-light, bpemb, bpemb-attention
    - **addresses** (List[Address], request body): List of addresses to parse.

    Returns:
    - **JSONResponse**: JSON response containing the parsed addresses, along with the model type and version.

    Raises:
    - **HTTPException (422)**: If the addresses parameter is empty or if the specified parsing model is not implemented.

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
    if not addresses:
        raise HTTPException(status_code=422, detail="Addresses parameter must not be empty")
    if parsing_model not in MODEL_MAPPING_CHOICES:
        raise HTTPException(
            status_code=422, detail=f"Parsing model not implemented, available choices: {list(MODEL_MAPPING_CHOICES)}"
        )
    return JSONResponse(content=resp)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
