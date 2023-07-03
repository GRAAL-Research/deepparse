"""REST API."""
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from deepparse.parser import AddressParser
from deepparse.cli.parser_arguments_adder import choices


app = FastAPI()


class Address(BaseModel):
    raw: str


@app.post("/parse/{parsing_model}")
async def parse(parsing_model: str, addresses: list[Address]):
    assert addresses, "Addresses parameter must not be empty"
    assert parsing_model in choices, f"Parsing model not implemented, available choices: {choices}"

    address_parser = AddressParser(model_type=parsing_model)
    parsed_addresses = address_parser([address.raw for address in addresses])

    response_payload = {
        "model_type": address_parser.get_formatted_model_name(),
        "version": address_parser.version,
        "parsed_addresses": {
            raw_address.raw: parsed_address.to_dict()
            for parsed_address, raw_address in zip(parsed_addresses, addresses)
        },
    }

    return JSONResponse(content=response_payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
