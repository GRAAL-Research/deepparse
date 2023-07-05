import pytest
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from deepparse.app.app import app, format_parsed_addresses


@pytest.fixture(scope="session", name="client")
def fixture_client():
    with TestClient(app) as _client:
        yield _client


def test_parse_addresses(client: TestClient):
    parsing_model = "bpemb"
    raw_address_1 = "2325 Rue de l'Université, Québec, QC G1V 0A6"
    raw_address_2 = "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"

    parsed_address_1 = {
        'EOS': None,
        'GeneralDelivery': None,
        'Municipality': 'Québec,',
        'Orientation': None,
        'PostalCode': 'G1V 0A6',
        'Province': 'QC',
        'StreetName': "Rue de l'Université",
        'StreetNumber': '2325',
        'Unit': None,
    }

    parsed_address_2 = {
        'EOS': None,
        'GeneralDelivery': None,
        'Municipality': 'Quebec city',
        'Orientation': 'Ouest',
        'PostalCode': 'G1L 1B6',
        'Province': 'Quebec',
        'StreetName': 'rue des Lilas',
        'StreetNumber': '350',
        'Unit': None,
    }

    expected_parsed_addresses = [{raw_address_1: parsed_address_1}, {raw_address_2: parsed_address_2}]

    json_addresses = jsonable_encoder(
        [
            {
                "raw": raw_address_1,
            },
            {"raw": raw_address_2},
        ]
    )

    model_version = "aa32fa918494b461202157c57734c374\n"

    mocked_response = JSONResponse(
        {
            "model_type": parsing_model,
            "parsed_addresses": expected_parsed_addresses,
            "version": model_version,
        }
    )

    def mock_format_parsed_addresses() -> JSONResponse:
        return mocked_response

    app.dependency_overrides[format_parsed_addresses] = mock_format_parsed_addresses

    response = client.post(f"/parse/{parsing_model}", json=json_addresses)

    assert response.status_code == 200
    assert response.json() == {
        "model_type": parsing_model,
        "parsed_addresses": expected_parsed_addresses,
        "version": model_version,
    }


def test_parse_empty_addresses(client: TestClient):
    with pytest.raises(AssertionError, match="Addresses parameter must not be empty"):
        client.post("/parse/bpemb", json=[])


def test_parse_invalid_model(client: TestClient):
    with pytest.raises(AssertionError):
        addresses = [
            {"raw": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
            {"raw": "2325 Rue de l'Université, Québec, QC G1V 0A6"},
        ]
        client.post("/parse/invalid_model", json=addresses)
