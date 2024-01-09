import os

from unittest.mock import Mock
from unittest import skipIf
import pytest
import httpx

try:
    from fastapi.testclient import TestClient
    from fastapi.encoders import jsonable_encoder
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Ensure you installed the packages for the app_requirements.txt file found in the root of the project"
    ) from e

from deepparse.app.api import api, get_address_parser_service, AddressParserService


@pytest.fixture(scope="session", name="client")
def fixture_client():
    with TestClient(api) as _client:
        yield _client


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test_parse(client: TestClient):
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
                "address": raw_address_1,
            },
            {"address": raw_address_2},
        ]
    )

    model_version = "aa32fa918494b461202157c57734c374"

    mocked_response = {
        "model_type": parsing_model,
        "parsed_addresses": expected_parsed_addresses,
        "version": model_version,
    }

    mocked_address_parser_service = Mock(spec=AddressParserService)
    mocked_address_parser_service.return_value = mocked_response
    api.dependency_overrides[get_address_parser_service] = lambda: mocked_address_parser_service

    response = client.post(f"/parse/{parsing_model}", json=json_addresses)
    del api.dependency_overrides[get_address_parser_service]

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
        mocked_address_parser_service = Mock(spec=AddressParserService)
        mocked_address_parser_service.address_parser_mapping = {"bpemb": "bpemb"}
        api.dependency_overrides[get_address_parser_service] = lambda: mocked_address_parser_service
        addresses = [
            {"address": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
            {"address": "2325 Rue de l'Université, Québec, QC G1V 0A6"},
        ]
        client.post("/parse/invalid_model", json=addresses)
    del api.dependency_overrides[get_address_parser_service]


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test__parse__should_parse_addresses__when_multiple_addresses(client: TestClient):
    model_type = "bpemb"

    addresses = [
        {"address": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
        {"address": "2325 Rue de l'Université, Québec, QC G1V 0A6"},
    ]
    response = client.post(f"/parse/{model_type}", json=addresses)

    assert httpx.codes.is_success(response.status_code)


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test__parse__should_parse_addresses__when_single_address(client: TestClient):
    model_type = "bpemb"

    addresses = [
        {"address": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
    ]
    response = client.post(f"/parse/{model_type}", json=addresses)

    assert httpx.codes.is_success(response.status_code)
