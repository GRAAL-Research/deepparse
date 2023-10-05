import os

from unittest import skipIf
import pytest
from fastapi.testclient import TestClient

from deepparse.app.app import app


@skipIf(os.environ["TEST_LEVEL"] == "unit", "Cannot run test without a proper GPU or RAM.")
@pytest.fixture(scope="session", name="client")
def fixture_client():
    with TestClient(app) as _client:
        yield _client
