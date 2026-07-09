import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings, load_settings

FAKE_ENV = {
    "CLOUDFLARE_TOKEN": "fake-token",
    "R2_ACCOUNT_ID": "myaccountid",
    "R2_ACCESS_KEY_ID": "fake-key",
    "R2_SECRET_ACCESS_KEY": "fake-secret",
    "MOTHERDUCK_TOKEN": "fake-md-token",
    "DAGSHUB_USERNAME": "user",
    "DAGSHUB_TOKEN": "token",
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
    "MODAL_TOKEN_ID": "id",
    "MODAL_TOKEN_SECRET": "secret",
}


@pytest.fixture(autouse=True)
def clear_cache() -> Iterator[None]:
    """Clear LRU cache before and after each test."""
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


def test_defaults() -> None:
    """Verify default configuration values."""
    with patch.dict(os.environ, FAKE_ENV, clear=True):
        s = load_settings()
        assert s.r2_bucket_name == "chicago-crimes-analytics"
        assert s.api_host == "0.0.0.0"
        assert s.api_port == 8000
        assert s.debug is False
        assert s.base_path == Path("data")
        assert s.models_path == Path("models")


def test_r2_endpoint_url() -> None:
    """Verify r2_endpoint_url is correctly constructed from account ID."""
    with patch.dict(os.environ, FAKE_ENV, clear=True):
        s = load_settings()
        assert s.r2_endpoint_url == "https://myaccountid.r2.cloudflarestorage.com"


def test_custom_r2_bucket_name() -> None:
    """Verify custom bucket name can be set via environment."""
    custom_env = {**FAKE_ENV, "R2_BUCKET_NAME": "custom-bucket"}
    with patch.dict(os.environ, custom_env, clear=True):
        s = load_settings()
        assert s.r2_bucket_name == "custom-bucket"


def test_custom_base_path() -> None:
    """Verify custom base path can be set via environment."""
    custom_env = {**FAKE_ENV, "BASE_PATH": "/custom/data"}
    with patch.dict(os.environ, custom_env, clear=True):
        s = load_settings()
        assert s.base_path == Path("/custom/data")
        assert s.raw_path == Path("/custom/data/raw")
        assert s.processed_path == Path("/custom/data/processed")
        assert s.checkpoints_path == Path("/custom/data/checkpoints")


def test_custom_models_path() -> None:
    """Verify custom models path can be set via environment."""
    custom_env = {**FAKE_ENV, "MODELS_PATH": "/custom/models"}
    with patch.dict(os.environ, custom_env, clear=True):
        s = load_settings()
        assert s.models_path == Path("/custom/models")


def test_custom_api_config() -> None:
    """Verify API host and port can be customized."""
    custom_env = {**FAKE_ENV, "API_HOST": "127.0.0.1", "API_PORT": "9000"}
    with patch.dict(os.environ, custom_env, clear=True):
        s = load_settings()
        assert s.api_host == "127.0.0.1"
        assert s.api_port == 9000


def test_debug_mode_enabled() -> None:
    """Verify debug mode can be enabled via environment."""
    custom_env = {**FAKE_ENV, "DEBUG": "true"}
    with patch.dict(os.environ, custom_env, clear=True):
        s = load_settings()
        assert s.debug is True


def test_invalid_port_raises() -> None:
    """Verify ValidationError for port out of range."""
    with (
        patch.dict(os.environ, {**FAKE_ENV, "API_PORT": "99999"}, clear=True),
        pytest.raises(ValidationError),
    ):
        Settings()  # type: ignore[call-arg]


def test_missing_required_field_raises() -> None:
    """Verify ValidationError when required field is missing."""
    missing_field_env = {k: v for k, v in FAKE_ENV.items() if k != "CLOUDFLARE_TOKEN"}
    with (
        patch.dict(os.environ, missing_field_env, clear=True),
        pytest.raises(ValidationError) as exc_info,
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert "cloudflare_token" in str(exc_info.value).lower()


def test_computed_paths_with_default_base() -> None:
    """Verify computed paths work correctly with default base path."""
    with patch.dict(os.environ, FAKE_ENV, clear=True):
        s = load_settings()
        assert s.raw_path == s.base_path / "raw"
        assert s.processed_path == s.base_path / "processed"
        assert s.checkpoints_path == s.base_path / "checkpoints"


def test_settings_uses_env_file() -> None:
    """Verify SettingsConfigDict is configured for .env file loading."""
    # This test verifies the configuration exists
    assert Settings.model_config.get("env_file") == ".env"
    assert Settings.model_config.get("env_file_encoding") == "utf-8"
    assert Settings.model_config.get("case_sensitive") is False


def test_load_settings_cache() -> None:
    """Verify load_settings uses caching (singleton behavior)."""
    with patch.dict(os.environ, FAKE_ENV, clear=True):
        s1 = load_settings()
        s2 = load_settings()

        # Same object due to caching
        assert s1 is s2


def test_load_settings_cache_clear() -> None:
    """Verify cache can be cleared to create new instance."""
    with patch.dict(os.environ, FAKE_ENV, clear=True):
        s1 = load_settings()
        load_settings.cache_clear()
        s2 = load_settings()

        # Different objects after cache clear
        assert s1 is not s2
        # But they have the same data
        assert s1.r2_bucket_name == s2.r2_bucket_name
