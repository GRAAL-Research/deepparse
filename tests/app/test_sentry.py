from unittest.mock import patch

import pytest

try:
    from deepparse.app import sentry as sentry_module

    APP_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    APP_DEPS_AVAILABLE = False


@pytest.mark.skipif(not APP_DEPS_AVAILABLE, reason="The app extra (sentry-sdk) is not installed.")
def test_configureSentry_usesConservativeSamplingAndDoesNotSendPII():
    # config(key, default, cast=...) -> we return the default so the test pins the safe defaults.
    def fake_config(key, default=None, cast=None):  # pylint: disable=unused-argument
        return default

    with patch.object(sentry_module, "sentry_sdk") as sentry_mock:
        with patch.object(sentry_module, "config", side_effect=fake_config):
            sentry_module.configure_sentry()

    _, init_kwargs = sentry_mock.init.call_args
    # Conservative sampling defaults (not 1.0) to limit volume and third-party data exposure.
    assert init_kwargs["traces_sample_rate"] == 0.1
    assert init_kwargs["profiles_sample_rate"] == 0.1
    # Parsed addresses are personal data: never ship request bodies or PII to Sentry.
    assert init_kwargs["max_request_body_size"] == "never"
    assert init_kwargs["send_default_pii"] is False
