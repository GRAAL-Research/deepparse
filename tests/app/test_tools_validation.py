import pytest

try:
    from fastapi import HTTPException

    from deepparse.app.tools import Address, format_parsed_addresses

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
