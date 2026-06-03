from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )
    # Cloudflare
    cloudflare_token: str = ""
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = ""
    r2_endpoint_url: str = ""

    # DasgHub / MLflow
    dagshub_username: str = ""
    dagshub_token: str = ""
    mlflow_tracking_uri: str = ""

    # Modal
    modal_token_id: str = ""
    modal_token_secret: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Paths locales
    base_path: Path = Path("data")
    models_path: Path = Path("models")
    checkpoints_path: Path = Path("data/checkpoints")


@lru_cache
def load_settings() -> Settings:
    return Settings()
