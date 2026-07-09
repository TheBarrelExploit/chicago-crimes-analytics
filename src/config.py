"""Configuración central de la aplicación usando Pydantic Settings.

Este módulo define el modelo de configuración que carga variables de entorno
desde un archivo .env o del sistema, con validación automática de tipos.

Las variables sensibles (tokens, keys) se manejan como SecretStr para evitar
exposición accidental en logs o tracebacks.

Usage:
    >>> from src.config import load_settings
    >>> settings = load_settings()
    >>> print(settings.api_port)
    8000
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la aplicación usando variables de entorno.

    Las variables se pueden definir en un archivo .env o como variables
    del sistema. Todas las variables requeridas deben estar presentes
    al momento de la instanciación.

    Attributes:
        cloudflare_token: Token de API de Cloudflare (secreto).
        r2_account_id: ID de cuenta de Cloudflare R2 (secreto).
        r2_access_key_id: Access key para R2 (secreto).
        r2_secret_access_key: Secret key para R2 (secreto).
        r2_bucket_name: Nombre del bucket R2 (default: chicago-crimes-analytics).
        api_host: Host para el servidor API (default: 0.0.0.0).
        api_port: Puerto para el servidor API (default: 8000).
        debug: Modo debug (default: False).
        base_path: Ruta base para almacenamiento local.
        models_path: Ruta para modelos entrenados.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Cloudflare R2 ---
    cloudflare_token: SecretStr | None = Field(default=None, description="Cloudflare API token")
    r2_account_id: SecretStr = Field(description="Cloudflare Account ID")
    r2_access_key_id: SecretStr = Field(description="R2 Access Key ID")
    r2_secret_access_key: SecretStr = Field(description="R2 Secret Access Key")
    r2_bucket_name: str = Field(
        default="chicago-crimes-analytics",
        description="Nombre del bucket R2",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def r2_endpoint_url(self) -> str:
        """Construye la URL del endpoint R2 a partir del Account ID.

        Returns:
            URL completa del endpoint R2 en formato:
            https://{account_id}.r2.cloudflarestorage.com
        """
        return (
            f"https://{self.r2_account_id.get_secret_value()}.r2.cloudflarestorage.com"
        )

    # --- DagsHub / MLflow ---
    dagshub_username: SecretStr = Field(description="Username DagsHub")
    dagshub_token: SecretStr = Field(description="Token DagsHub")
    mlflow_tracking_uri: SecretStr = Field(description="MLflow tracking URI en DagsHub")

    # --- MotherDuck ---
    motherduck_token: SecretStr | None = Field(default=None, description="MotherDuck authentication token")
    motherduck_database: str = Field(
        default="chicago_crimes",
        description="MotherDuck database name",
    )

    # --- Modal ---
    modal_token_id: SecretStr | None = Field(default=None, description="Token ID Modal")
    modal_token_secret: SecretStr | None = Field(default=None, description="Secret Token Modal")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", description="Host for the API server")
    api_port: int = Field(
        default=8000, ge=1, le=65535, description="Port for the API server"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # --- Paths locales ---
    base_path: Path = Field(
        default=Path("data"), description="Base path for local storage"
    )
    models_path: Path = Field(
        default=Path("models"), description="Path for trained models"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpoints_path(self) -> Path:
        """Ruta para almacenar checkpoints de modelos durante entrenamiento.

        Returns:
            Path: Ruta completa a data/checkpoints
        """
        return self.base_path / "checkpoints"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_path(self) -> Path:
        """Ruta para almacenar datos crudos sin procesar.

        Returns:
            Path: Ruta completa a data/raw
        """
        return self.base_path / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_path(self) -> Path:
        """Ruta para almacenar datos procesados y limpios.

        Returns:
            Path: Ruta completa a data/processed
        """
        return self.base_path / "processed"


@lru_cache
def load_settings() -> Settings:
    """Carga y cachea la configuración de la aplicación.

    Esta función utiliza lru_cache para que la configuración se cargue
    una sola vez y se reutilice en toda la aplicación.

    Returns:
        Settings: Instancia única (singleton) de la configuración validada.

    Example:
        >>> settings = load_settings()
        >>> print(settings.api_port)
        8000

    Note:
        Si alguna variable requerida no está presente en el entorno o
        archivo .env, Pydantic lanzará un error de validación,
        para generar test se deberia utilizar load_settings.cache_clear()
    """
    return Settings()
