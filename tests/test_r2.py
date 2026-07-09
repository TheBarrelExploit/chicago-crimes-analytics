"""Tests for R2BucketClient.

Estrategia: ningún test toca R2 real. El cliente boto3 se reemplaza
con un MagicMock, lo que permite verificar comportamiento sin red ni
credenciales reales.
"""

import io
import os
from pathlib import Path
from typing import IO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.config import Settings
from src.storage.r2 import R2BucketClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_settings() -> Settings:
    """Settings válidos construidos desde variables de entorno mockeadas.

    Usa patch.dict para inyectar los valores requeridos sin tocar .env real.
    """
    env_vars = {
        "CLOUDFLARE_TOKEN": "fake-cf-token",
        "R2_ACCOUNT_ID": "fake-account-id",
        "R2_ACCESS_KEY_ID": "fake-access-key",
        "R2_SECRET_ACCESS_KEY": "fake-secret-key",
        "MOTHERDUCK_TOKEN": "fake-md-token",
        "DAGSHUB_USERNAME": "fake-user",
        "DAGSHUB_TOKEN": "fake-dagshub-token",
        "MLFLOW_TRACKING_URI": "http://fake-mlflow",
        "MODAL_TOKEN_ID": "fake-modal-id",
        "MODAL_TOKEN_SECRET": "fake-modal-secret",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        return Settings()  # type: ignore[call-arg]


@pytest.fixture()
def mock_boto3_client() -> MagicMock:
    """Cliente boto3 falso que no hace llamadas de red."""
    return MagicMock()


@pytest.fixture()
def r2_client(fake_settings: Settings, mock_boto3_client: MagicMock) -> R2BucketClient:
    """R2BucketClient con el cliente boto3 ya inyectado como mock.

    Inyectamos directamente en _client para evitar que boto3.client()
    intente conectarse a R2 real.
    """
    client = R2BucketClient(fake_settings)
    client._client = mock_boto3_client  # bypass lazy init
    return client


# ---------------------------------------------------------------------------
# _prepare_extra_arg
# ---------------------------------------------------------------------------


class TestPrepareExtraArg:
    """Tests para la lógica de construcción de ExtraArgs."""

    def test_returns_none_when_no_args(self, r2_client: R2BucketClient) -> None:
        """Sin argumentos debe retornar None, no un dict vacío."""
        result = r2_client._prepare_extra_arg()
        assert result is None

    def test_explicit_content_type(self, r2_client: R2BucketClient) -> None:
        """content_type explícito debe aparecer en ContentType."""
        result = r2_client._prepare_extra_arg(content_type="application/json")
        assert result is not None
        assert result["ContentType"] == "application/json"

    def test_infers_content_type_from_parquet(self, r2_client: R2BucketClient) -> None:
        """Extensión .parquet debe inferir application/parquet."""
        result = r2_client._prepare_extra_arg(filename="data.parquet")
        assert result is not None
        assert result["ContentType"] == "application/parquet"

    def test_infers_content_type_from_json(self, r2_client: R2BucketClient) -> None:
        result = r2_client._prepare_extra_arg(filename="model.json")
        assert result is not None
        assert result["ContentType"] == "application/json"

    def test_infers_content_type_from_csv(self, r2_client: R2BucketClient) -> None:
        result = r2_client._prepare_extra_arg(filename="data.csv")
        assert result is not None
        assert result["ContentType"] == "text/csv"

    def test_explicit_content_type_overrides_filename(
        self, r2_client: R2BucketClient
    ) -> None:
        """content_type explícito tiene prioridad sobre la extensión del archivo."""
        result = r2_client._prepare_extra_arg(
            content_type="application/octet-stream",
            filename="data.parquet",
        )
        assert result is not None
        assert result["ContentType"] == "application/octet-stream"

    def test_unknown_extension_returns_none(self, r2_client: R2BucketClient) -> None:
        """Extensión desconocida sin content_type explícito retorna None."""
        result = r2_client._prepare_extra_arg(filename="file.xyz")
        assert result is None

    def test_metadata_only(self, r2_client: R2BucketClient) -> None:
        """Solo metadata debe retornar dict con Metadata."""
        result = r2_client._prepare_extra_arg(metadata={"source": "api"})
        assert result is not None
        assert result["Metadata"] == {"source": "api"}

    def test_content_type_and_metadata_together(
        self, r2_client: R2BucketClient
    ) -> None:
        result = r2_client._prepare_extra_arg(
            content_type="text/csv",
            metadata={"year": "2024"},
        )
        assert result is not None
        assert result["ContentType"] == "text/csv"
        assert result["Metadata"] == {"year": "2024"}


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------


class TestUploadFile:
    """Tests para subida de archivos locales."""

    def test_raises_if_file_not_found(self, r2_client: R2BucketClient) -> None:
        """Archivo inexistente debe lanzar FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            r2_client.upload_file(
                local_path="nonexistent.parquet", key="raw/data.parquet"
            )

    def test_raises_if_path_is_directory(
        self, r2_client: R2BucketClient, tmp_path: Path
    ) -> None:
        """Directorio en lugar de archivo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Path is not a file"):
            r2_client.upload_file(local_path=tmp_path, key="raw/data.parquet")

    def test_successful_upload(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Upload exitoso debe llamar boto3 con los parámetros correctos."""
        test_file = tmp_path / "crimes.parquet"
        test_file.write_bytes(b"fake parquet data")

        r2_client.upload_file(local_path=test_file, key="raw/crimes.parquet")

        mock_boto3_client.upload_file.assert_called_once()
        call_kwargs = mock_boto3_client.upload_file.call_args.kwargs
        assert call_kwargs["Filename"] == str(test_file)
        assert call_kwargs["Key"] == "raw/crimes.parquet"

    def test_upload_infers_content_type(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Debe inferir ContentType desde la extensión del archivo."""
        test_file = tmp_path / "data.json"
        test_file.write_bytes(b'{"key": "value"}')

        r2_client.upload_file(local_path=test_file, key="models/data.json")

        call_kwargs = mock_boto3_client.upload_file.call_args.kwargs
        assert call_kwargs["ExtraArgs"] == {"ContentType": "application/json"}

    def test_upload_with_metadata(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Metadata debe pasarse correctamente a boto3."""
        test_file = tmp_path / "model.json"
        test_file.write_bytes(b"{}")
        meta = {"version": "1.0", "accuracy": "0.95"}

        r2_client.upload_file(
            local_path=test_file,
            key="models/model.json",
            metadata=meta,
        )

        call_kwargs = mock_boto3_client.upload_file.call_args.kwargs
        assert call_kwargs["ExtraArgs"]["Metadata"] == meta

    def test_upload_propagates_client_error(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """ClientError de boto3 debe propagarse sin suprimirse."""
        test_file = tmp_path / "data.csv"
        test_file.write_bytes(b"a,b,c")

        mock_boto3_client.upload_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "upload_file",
        )

        with pytest.raises(ClientError):
            r2_client.upload_file(local_path=test_file, key="raw/data.csv")


# ---------------------------------------------------------------------------
# upload_bytes
# ---------------------------------------------------------------------------


class TestUploadBytes:
    """Tests para subida de datos en memoria."""

    def test_raises_on_empty_bytes(self, r2_client: R2BucketClient) -> None:
        """Bytes vacíos deben lanzar ValueError."""
        with pytest.raises(ValueError, match="Cannot upload empty bytes"):
            r2_client.upload_bytes(b"", key="test/empty.txt")

    def test_raises_on_invalid_type(self, r2_client: R2BucketClient) -> None:
        """Tipo inválido debe lanzar TypeError."""
        with pytest.raises(TypeError, match="Expected bytes or file-like"):
            r2_client.upload_bytes(12345, key="test/invalid.txt")  # type: ignore[arg-type]

    def test_upload_bytes_object(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
    ) -> None:
        """Bytes debe convertirse a BytesIO antes de llamar upload_fileobj."""
        r2_client.upload_bytes(b"hello r2", key="test/hello.txt")

        mock_boto3_client.upload_fileobj.assert_called_once()
        call_args = mock_boto3_client.upload_fileobj.call_args
        fileobj = call_args.args[0]
        # Verifica que se pasó un objeto file-like con el contenido correcto
        assert isinstance(fileobj, io.BytesIO)
        assert fileobj.read() == b"hello r2"

    def test_upload_file_like_object(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
    ) -> None:
        """IO[bytes] debe pasarse directamente sin envolver en BytesIO."""
        buffer: IO[bytes] = io.BytesIO(b"buffer data")

        r2_client.upload_bytes(buffer, key="test/buffer.txt")

        mock_boto3_client.upload_fileobj.assert_called_once()
        call_args = mock_boto3_client.upload_fileobj.call_args
        assert call_args.args[0] is buffer  # mismo objeto, no una copia

    def test_upload_bytes_with_content_type(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
    ) -> None:
        """content_type debe pasarse en ExtraArgs."""
        r2_client.upload_bytes(
            b"data",
            key="test/data.txt",
            content_type="text/plain",
        )

        call_kwargs = mock_boto3_client.upload_fileobj.call_args.kwargs
        assert call_kwargs["ExtraArgs"]["ContentType"] == "text/plain"

    def test_upload_bytes_no_extra_args_passes_none(
        self,
        r2_client: R2BucketClient,
        mock_boto3_client: MagicMock,
    ) -> None:
        """Sin content_type ni metadata, ExtraArgs debe ser None."""
        r2_client.upload_bytes(b"data", key="test/data.bin")

        call_kwargs = mock_boto3_client.upload_fileobj.call_args.kwargs
        assert call_kwargs["ExtraArgs"] is None


# ---------------------------------------------------------------------------
# shutdown y lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Tests para shutdown y context manager."""

    def test_shutdown_sets_flag(self, r2_client: R2BucketClient) -> None:
        """shutdown() debe marcar el cliente como cerrado."""
        r2_client.shutdown()
        assert r2_client._is_shutdown is True

    def test_shutdown_idempotent(self, r2_client: R2BucketClient) -> None:
        """shutdown() llamado dos veces no debe lanzar error."""
        r2_client.shutdown()
        r2_client.shutdown()  # segunda llamada segura

    def test_client_raises_after_shutdown(self, r2_client: R2BucketClient) -> None:
        """Acceder a .client después de shutdown() debe lanzar RuntimeError."""
        r2_client.shutdown()
        with pytest.raises(RuntimeError, match="already shut down"):
            _ = r2_client.client

    def test_context_manager_calls_shutdown(
        self, fake_settings: Settings, mock_boto3_client: MagicMock
    ) -> None:
        """El context manager debe llamar shutdown() al salir."""
        with R2BucketClient(fake_settings) as client:
            client._client = mock_boto3_client

        assert client._is_shutdown is True

    def test_context_manager_calls_shutdown_on_exception(
        self, fake_settings: Settings, mock_boto3_client: MagicMock
    ) -> None:
        """shutdown() debe llamarse incluso si ocurre una excepción dentro del with."""
        client_ref: R2BucketClient | None = None

        with pytest.raises(ValueError), R2BucketClient(fake_settings) as client:
            client._client = mock_boto3_client
            client_ref = client
            raise ValueError("something went wrong")

        assert client_ref is not None
        assert client_ref._is_shutdown is True


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    """Tests para la representación segura del cliente."""

    def test_repr_shows_bucket_name(self, r2_client: R2BucketClient) -> None:
        """repr() debe mostrar el nombre del bucket."""
        result = repr(r2_client)
        assert "chicago-crimes-analytics" in result

    def test_repr_does_not_expose_credentials(
        self, r2_client: R2BucketClient, fake_settings: Settings
    ) -> None:
        """repr() nunca debe exponer access key ni secret key."""
        result = repr(r2_client)
        assert fake_settings.r2_access_key_id.get_secret_value() not in result
        assert fake_settings.r2_secret_access_key.get_secret_value() not in result
