import os

from unittest import skipIf
import pytest
from fastapi.testclient import TestClient

from deepparse.app.api import api


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
@pytest.fixture(scope="session", name="client")
def fixture_client():
    with TestClient(api) as _client:
        yield _client
