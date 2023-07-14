"""Sentry configuration."""

from importlib.metadata import version as get_version
import sentry_sdk
from decouple import config


def configure_sentry() -> None:
    """Configure Sentry."""
    sentry_dsn = config("SENTRY_DSN", "")
    environment = config("ENVIRONMENT", "local")
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        release=f"deepparse@{get_version('deepparse')}",
        profiles_sample_rate=1.0,
        environment=environment,
        request_bodies="small",
    )
