from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi import HTTPException

    from deepparse.app import tools as app_tools
    from deepparse.app.tools import Address, format_parsed_addresses
    from deepparse.parser import FormattedParsedAddress

    APP_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    APP_DEPS_AVAILABLE = False


# These validations run before any model is loaded, so they are unit-testable without a GPU or downloaded
# weights. format_parsed_addresses is used as a FastAPI dependency: it must raise HTTPException (-> 422), not
# ValueError (which FastAPI would surface as a 500).
@pytest.mark.skipif(not APP_DEPS_AVAILABLE, reason="The app extra (fastapi) is not installed.")
def test_givenEmptyAddresses_whenFormatParsedAddresses_thenRaisesHTTPException422():
    with pytest.raises(HTTPException) as exception_info:
        format_parsed_addresses("bpemb", [])

    assert exception_info.value.status_code == 422


@pytest.mark.skipif(not APP_DEPS_AVAILABLE, reason="The app extra (fastapi) is not installed.")
def test_givenInvalidModel_whenFormatParsedAddresses_thenRaisesHTTPException422():
    with pytest.raises(HTTPException) as exception_info:
        format_parsed_addresses("not_a_real_model", [Address(raw="an address")])

    assert exception_info.value.status_code == 422


@pytest.mark.skipif(not APP_DEPS_AVAILABLE, reason="The app extra (fastapi) is not installed.")
def test_givenTooManyAddresses_whenFormatParsedAddresses_thenRaisesHTTPException413():
    # Guard against resource-exhaustion (DoS): a request over the limit must be rejected before any parsing.
    with patch.object(app_tools, "MAX_ADDRESSES_PER_REQUEST", 2):
        with pytest.raises(HTTPException) as exception_info:
            format_parsed_addresses("bpemb", [Address(raw="a"), Address(raw="b"), Address(raw="c")])

    assert exception_info.value.status_code == 413


@pytest.mark.skipif(not APP_DEPS_AVAILABLE, reason="The app extra (fastapi) is not installed.")
def test_givenDuplicateRawAddresses_whenFormatParsedAddresses_thenBothAreKept():
    # Two identical raw addresses must both appear in the response. Keying the response on the raw text (the
    # previous behaviour) collapsed them into a single entry, silently losing data.
    model_type = "bpemb"
    raw_address = "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"
    parsed = FormattedParsedAddress({raw_address: [("350", "StreetNumber"), ("rue des Lilas", "StreetName")]})

    parser_mock = MagicMock()
    parser_mock.model_type = model_type
    parser_mock.version = "a_version"
    parser_mock.return_value = [parsed, parsed]

    response = format_parsed_addresses(
        model_type, [Address(raw=raw_address), Address(raw=raw_address)], {model_type: parser_mock}
    )

    assert len(response["parsed_addresses"]) == 2
