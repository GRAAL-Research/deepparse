"""Sentry configuration."""

from importlib.metadata import version as get_version

import sentry_sdk
from decouple import config


def configure_sentry() -> None:
    """Configure Sentry.

    Sampling rates default to a conservative 0.1 and are overridable via the ``SENTRY_TRACES_SAMPLE_RATE`` and
    ``SENTRY_PROFILES_SAMPLE_RATE`` environment variables. Request bodies are never captured and PII sending is
    off, since parsed addresses are personal data and must not be shipped to a third party.
    """
    sentry_dsn = config("SENTRY_DSN", "")
    environment = config("ENVIRONMENT", "local")
    traces_sample_rate = config("SENTRY_TRACES_SAMPLE_RATE", 0.1, cast=float)
    profiles_sample_rate = config("SENTRY_PROFILES_SAMPLE_RATE", 0.1, cast=float)
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=traces_sample_rate,
        release=f"deepparse@{get_version('deepparse')}",
        profiles_sample_rate=profiles_sample_rate,
        environment=environment,
        max_request_body_size="never",
        send_default_pii=False,
    )
