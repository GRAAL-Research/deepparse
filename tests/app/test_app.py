from typing import Dict, List, Union
import os

from unittest.mock import MagicMock
from unittest import skipIf
import pytest

try:
    from fastapi.testclient import TestClient
    from fastapi.encoders import jsonable_encoder
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Ensure you installed the packages for the app_requirements.txt file found in the root of the project"
    ) from e

from deepparse.app.app import app, format_parsed_addresses, Address, AddressParser
from deepparse.parser import FormattedParsedAddress


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
@pytest.fixture(scope="session", name="client")
def fixture_client():
    with TestClient(app) as _client:
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
                "raw": raw_address_1,
            },
            {"raw": raw_address_2},
        ]
    )

    model_version = "aa32fa918494b461202157c57734c374\n"

    mocked_response = {
        "model_type": parsing_model,
        "parsed_addresses": expected_parsed_addresses,
        "version": model_version,
    }

    def mock_format_parsed_addresses() -> Dict[str, Union[str, List[Dict]]]:
        return mocked_response

    app.dependency_overrides[format_parsed_addresses] = mock_format_parsed_addresses

    response = client.post(f"/parse/{parsing_model}", json=json_addresses)

    assert response.status_code == 200
    assert response.json() == {
        "model_type": parsing_model,
        "parsed_addresses": expected_parsed_addresses,
        "version": model_version,
    }


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test_parse_empty_addresses(client: TestClient):
    with pytest.raises(AssertionError, match="Addresses parameter must not be empty"):
        client.post("/parse/bpemb", json=[])


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test_parse_invalid_model(client: TestClient):
    with pytest.raises(AssertionError):
        addresses = [
            {"raw": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
            {"raw": "2325 Rue de l'Université, Québec, QC G1V 0A6"},
        ]
        client.post("/parse/invalid_model", json=addresses)


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test_format_parsed_addresses():
    # Create a mock for the AddressParser
    model_type = "bpemb"
    version = "1234"

    address_parser_mock = MagicMock(spec=AddressParser)
    address_parser_mock.model_type = model_type
    address_parser_mock.version = version

    raw_address_1 = "2325 Rue de l'Université, Québec, QC G1V 0A6"
    raw_address_2 = "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"

    # Mock the return value of the address_parser method
    formatted_parsed_address_1 = FormattedParsedAddress(
        {
            raw_address_1: [
                ("Québec", "Municipality"),
                ("G1V 0A6", "PostalCode"),
                ("QC", "Province"),
                ("Rue de l'Université", "StreetName"),
                ("2325", "StreetNumber"),
            ]
        }
    )

    formatted_parsed_address_2 = FormattedParsedAddress(
        {
            raw_address_2: [
                ("350", "StreetNumber"),
                ("rue des Lilas", "StreetName"),
                ("Ouest", "Orientation"),
                ("Québec", "Municipality"),
                ("Québec", "Province"),
                ("G1L 1B6", "PostalCode"),
            ]
        }
    )

    formatted_parsed_addresses = [formatted_parsed_address_1, formatted_parsed_address_2]

    address_parser_mock.return_value = formatted_parsed_addresses

    # Use mocker.patch to replace the AddressParser instance with the mock
    address_parser_mapping = {"bpemb": address_parser_mock}
    # mocker.patch('deepparse.app.app.AddressParser', return_value=address_parser_mock)

    addresses = [Address(raw=raw_address_1), Address(raw=raw_address_2)]

    response = format_parsed_addresses(model_type, addresses, address_parser_mapping)

    # Assertions or checks on the response
    parsed_address_1 = formatted_parsed_address_1.to_dict()
    parsed_address_2 = formatted_parsed_address_2.to_dict()

    expected_parsed_addresses = {raw_address_1: parsed_address_1, raw_address_2: parsed_address_2}

    expected_response = {"model_type": model_type, "parsed_addresses": expected_parsed_addresses, "version": version}

    assert response == expected_response


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
def test_format_parsed_addresses__one_address():
    # Create a mock for the AddressParser
    model_type = "bpemb"
    version = "1234"

    address_parser_mock = MagicMock(spec=AddressParser)
    address_parser_mock.model_type = model_type
    address_parser_mock.version = version

    raw_address_1 = "2325 Rue de l'Université, Québec, QC G1V 0A6"

    # Mock the return value of the address_parser method
    formatted_parsed_address = FormattedParsedAddress(
        {
            raw_address_1: [
                ("Québec", "Municipality"),
                ("G1V 0A6", "PostalCode"),
                ("QC", "Province"),
                ("Rue de l'Université", "StreetName"),
                ("2325", "StreetNumber"),
            ]
        }
    )

    address_parser_mock.return_value = formatted_parsed_address

    addresses = [
        Address(raw=raw_address_1),
    ]

    address_parser_mapping = {"bpemb": address_parser_mock}

    response = format_parsed_addresses(model_type, addresses, address_parser_mapping)

    # Assertions or checks on the response
    parsed_address = formatted_parsed_address.to_dict()

    expected_parsed_address = {
        raw_address_1: parsed_address,
    }

    expected_response = {"model_type": model_type, "parsed_addresses": expected_parsed_address, "version": version}

    assert response == expected_response
