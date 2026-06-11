"""Cliente para almacenamiento en Cloudflare R2 (S3-compatible).

Este módulo proporciona un cliente para interactuar con buckets de Cloudflare R2,
permitiendo operaciones de upload de archivos locales y datos en memoria
con soporte para metadatos, tipos MIME y manejo robusto de errores.

Usage:
    >>> from src.config import load_settings
    >>> from src.storage.r2 import R2BucketClient
    >>> settings = load_settings()
    >>> client = R2BucketClient(settings)
    >>> client.upload_file(local_path="data.parquet", key="raw/data.parquet")
"""

import io
import logging
import types
from pathlib import Path
from typing import IO, NotRequired, TypedDict

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from src.config import Settings

logger = logging.getLogger(__name__)


class S3ExtrasArgs(TypedDict):
    """ExtraArgs para operaciones de upload en S3/R2.

    Este TypedDict define los argumentos adicionales que se pueden pasar
    a los métodos de upload de boto3 para controlar el comportamiento
    del objeto almacenado.

    Attributes:
        ContentType: Tipo MIME del contenido (ej: "application/json").
        Metadata: Diccionario con metadatos personalizados (key-value strings).

    Example:
        >>> args: S3ExtrasArgs = {
        ...     "ContentType": "application/parquet",
        ...     "Metadata": {"source": "chicago-police", "year": "2024"},
        ... }
    """

    ContentType: NotRequired[str]
    Metadata: NotRequired[dict[str, str]]


class R2BucketClient:
    """Cliente para interactuar con Cloudflare R2 (S3-compatible storage).

    Esta clase proporciona métodos para subir archivos y datos en memoria
    a buckets de Cloudflare R2, con soporte para metadatos, tipos MIME,
    y manejo robusto de errores.

    Attributes:
        settings: Configuración de la aplicación (validada por Pydantic).
        _bucket: Nombre del bucket R2 destino.
        _client: Cliente boto3 (lazy initialization).
        _is_shutdown: Flag indicando si el cliente fue cerrado.

    Example:
        >>> from src.config import load_settings
        >>> settings = load_settings()
        >>> client = R2BucketClient(settings)
        >>> # Subir archivo
        >>> client.upload_file(
        ...     local_path="data/crimes.parquet",
        ...     key="raw/crimes_2024.parquet",
        ...     metadata={"source": "chicago-police", "year": "2024"},
        ... )
        >>> # Subir bytes
        >>> client.upload_bytes(data=b"Hello R2", key="test/hello.txt", content_type="text/plain")
        >>> # Cerrar cliente
        >>> client.shutdown()
    """

    def __init__(self, settings: Settings):
        """Inicializa el cliente R2 con la configuración proporcionada.

        Args:
            settings: Configuración de la aplicación cargada desde variables
                de entorno o archivo .env mediante load_settings().

        Example:
            >>> from src.config import load_settings
            >>> settings = load_settings()  # Carga automática desde .env
            >>> client = R2BucketClient(settings)

        Note:
            La configuración es validada automáticamente por Pydantic al cargarla.
            No es necesario crear Settings manualmente.
        """
        self._settings: Settings = settings
        self._is_shutdown: bool = False
        self._client: S3Client | None = None
        self._bucket: str = self._settings.r2_bucket_name

    @property
    def client(self) -> S3Client:
        """Obtiene el cliente boto3 inicializado de forma lazy.

        El cliente se crea solo la primera vez que se accede a esta propiedad.
        Si el cliente ya fue cerrado (shutdown), lanza un error.

        Returns:
            S3Client: Cliente boto3 configurado para R2.

        Raises:
            RuntimeError: Si el cliente ya fue cerrado mediante shutdown().

        Example:
            >>> client = r2_client.client  # Primera vez: crea el cliente
            >>> client = r2_client.client  # Segunda vez: reutiliza el mismo
        """
        if self._is_shutdown:
            raise RuntimeError("R2 client is already shut down")

        if self._client is None:
            # Los SecretStr se convierten a string con .get_secret_value()

            self._client = boto3.client(
                service_name="s3",
                endpoint_url=self._settings.r2_endpoint_url,
                aws_access_key_id=self._settings.r2_access_key_id.get_secret_value(),
                aws_secret_access_key=self._settings.r2_secret_access_key.get_secret_value(),
                region_name="auto",
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        return self._client

    def _prepare_extra_arg(
        self,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        filename: str | Path | None = None,
    ) -> S3ExtrasArgs | None:
        """Prepara los argumentos adicionales para upload a S3/R2.

        Este método construye el diccionario de ExtraArgs basado en los
        parámetros proporcionados. Infiere automáticamente el ContentType
        desde la extensión del archivo si no se especifica explícitamente.

        Args:
            content_type: Tipo MIME del contenido. Si se proporciona, tiene
                prioridad sobre la inferencia por extensión.
            metadata: Diccionario con metadatos personalizados. Cada clave
                y valor debe ser string.
            filename: Nombre o ruta del archivo. Se usa para inferir el
                ContentType basado en la extensión.

        Returns:
            Diccionario con ExtraArgs listo para boto3, o None si no hay
            argumentos que pasar.

        Example:
            >>> client._prepare_extra_arg(content_type="application/json", metadata={"version": "1.0"})
            {'ContentType': 'application/json', 'Metadata': {'version': '1.0'}}

            >>> client._prepare_extra_arg(filename="data.parquet")
            {'ContentType': 'application/parquet'}
        """
        extra_args: S3ExtrasArgs = {}

        if content_type:
            extra_args["ContentType"] = content_type
        elif filename:
            ext = Path(filename).suffix.lower()
            content_type_map = {
                ".parquet": "application/parquet",
                ".json": "application/json",
                ".csv": "text/csv",
            }
            if ext in content_type_map:
                extra_args["ContentType"] = content_type_map[ext]

        if metadata:
            extra_args["Metadata"] = metadata
        return extra_args if extra_args else None

    def upload_file(
        self,
        *,
        local_path: str | Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Sube un archivo local al bucket R2.

        Este método lee un archivo del sistema de archivos local y lo sube
        al bucket R2 configurado. Soporta metadatos personalizados y
        detección automática del tipo MIME por extensión.

        Args:
            local_path: Ruta del archivo en el sistema local. Puede ser
                string o objeto Path.
            key: Clave (ruta) destino dentro del bucket. Ejemplo:
                "raw/data/crimes_2024.parquet" o "models/xgboost_v1.json".
            content_type: Tipo MIME del archivo. Si no se especifica, se
                infiere de la extensión del archivo (.parquet, .json, .csv).
            metadata: Diccionario con metadatos personalizados. Útil para
                trazabilidad (ej: {"source": "api", "year": "2024"}).

        Raises:
            FileNotFoundError: Si el archivo local no existe.
            ValueError: Si la ruta proporcionada no es un archivo.
            ClientError: Si ocurre un error durante la subida a R2
                (autenticación, red, permisos, etc.).
            RuntimeError: Si el cliente ya fue cerrado (shutdown).

        Example:
            >>> # Subida básica
            >>> client.upload_file(local_path="data/crimes.parquet", key="raw/crimes_2024.parquet")

            >>> # Subida con metadatos
            >>> client.upload_file(
            ...     local_path="models/model.json",
            ...     key="models/xgboost_v1.json",
            ...     content_type="application/json",
            ...     metadata={"version": "1.0.0", "accuracy": "0.95", "pipeline_id": "pipeline_001"},
            ... )

        Note:
            Los metadatos solo pueden contener valores string. Si necesitas
            otros tipos, conviértelos a string antes de pasarlos.
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if not local_path.is_file():
            raise ValueError(f"Path is not a file: {local_path}")
        extra: S3ExtrasArgs | None = self._prepare_extra_arg(
            content_type=content_type, metadata=metadata, filename=local_path
        )

        logger.info("Uploading %s -> s3://%s/%s", local_path, self._bucket, key)
        try:
            self.client.upload_file(
                Filename=str(local_path),
                Bucket=self._bucket,
                Key=key,
                ExtraArgs={**extra} or None,
            )
            logger.info("Upload OK: %s", key)
        except ClientError as e:
            logger.error("Error file upload key: %s error: %s", key, e)
            raise

    def upload_bytes(
        self,
        data: bytes | IO[bytes],
        key: str,
        metadata: dict[str, str] | None = None,
        content_type: str | None = None,
        filename: str | Path | None = None,
    ) -> None:
        """Sube datos en memoria (bytes o file-like) directamente a R2.

        Este método es útil para datos generados en tiempo real sin necesidad
        de escribirlos primero a disco. Ideal para APIs, streaming, o
        transformaciones en memoria.

        Args:
            data: Datos a subir. Puede ser bytes o cualquier objeto file-like
                que implemente el método read() (ej: BytesIO).
            key: Clave destino en el bucket.
            content_type: Tipo MIME del contenido. Si no se especifica y se
                proporciona filename, se infiere de la extensión.
            metadata: Diccionario con metadatos personalizados.
            filename: Nombre de archivo opcional para inferir ContentType
                cuando no se especifica explícitamente.

        Raises:
            ValueError: Si los datos son bytes vacíos.
            TypeError: Si data no es bytes ni file-like.
            ClientError: Si falla la subida a R2.
            RuntimeError: Si el cliente ya fue cerrado (shutdown).

        Example:
            >>> # Subir bytes directamente
            >>> client.upload_bytes(
            ...     data=b"Hello, Cloudflare R2!", key="test/hello.txt", content_type="text/plain"
            ... )

            >>> # Subir JSON generado en memoria
            >>> import json
            >>> data = json.dumps({"status": "ok", "count": 100}).encode()
            >>> client.upload_bytes(
            ...     data=data,
            ...     key="api/response.json",
            ...     content_type="application/json",
            ...     metadata={"timestamp": "2024-01-15T10:30:00Z"},
            ... )

            >>> # Subir desde un buffer
            >>> import io
            >>> buffer = io.BytesIO(b"some data")
            >>> client.upload_bytes(
            ...     data=buffer,
            ...     key="test/buffer.txt",
            ...     filename="buffer.txt",  # Para inferir ContentType
            ... )

        Note:
            Este método convierte automáticamente bytes a BytesIO internamente.
            Si pasas un objeto file-like, se usa directamente sin copiar.
        """
        if isinstance(data, bytes):
            if not data:
                raise ValueError("Cannot upload empty bytes")
            fileobj = io.BytesIO(data)
        elif hasattr(data, "read"):
            fileobj = data
        else:
            raise TypeError(f"Expected bytes or file-like, got {type(data)}")

        extra: S3ExtrasArgs | None = self._prepare_extra_arg(
            content_type=content_type, metadata=metadata, filename=filename
        )

        logger.info(
            "Uploading in-memory object -> s3://%s/%s",
            self._bucket,
            key,
        )
        try:
            self.client.upload_fileobj(
                fileobj, self._bucket, key, ExtraArgs={**extra} or None
            )
            logger.info("Upload OK: %s", key)
        except ClientError as e:
            logger.error("Error bytes upload key: %s error: %s", key, e)
            raise

    def shutdown(self) -> None:
        """Cierra el cliente liberando recursos de red.

        Este método debe llamarse cuando la aplicación se apaga para
        liberar conexiones HTTP y evitar warnings de recursos no cerrados.

        Es seguro llamar shutdown() múltiples veces; las llamadas adicionales
        no tienen efecto.

        Example:
            >>> client = R2BucketClient(settings)
            >>> try:
            ...     client.upload_file(local_path="data.parquet", key="data.parquet")
            ... finally:
            ...     client.shutdown()  # Siempre cerrar

        Note:
            Después de llamar a shutdown(), cualquier intento de usar el
            cliente lanzará RuntimeError.
        """
        if self._is_shutdown:
            return

        logger.info("Shutting down R2 client...")

        if self._client is not None:
            self._client.close()
            self._client = None

        self._is_shutdown = True
        logger.info("R2 client shutdown complete")

    def __repr__(self) -> str:
        """Return string representation of the R2BucketClient.

        Returns:
            String showing the bucket name for debugging purposes.

        Example:
            >>> client = R2BucketClient(settings)
            >>> repr(client)
            "R2BucketClient(bucket='test-bucket')"
        """
        return f"R2BucketClient(bucket={self._bucket!r})"

    def __enter__(self) -> "R2BucketClient":
        """Entra en el context manager.

        Returns:
            La instancia del cliente para usar dentro del bloque.

        Example:
            >>> with R2BucketClient(settings) as client:
            ...     client.upload_file(local_path="data.parquet", key="data.parquet")
            ... # shutdown() se llama automáticamente
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Sale del context manager y cierra el cliente.

        Args:
            exc_type: Tipo de excepción si ocurrió alguna.
            exc_val: Valor de la excepción si ocurrió alguna.
            exc_tb: Traceback de la excepción si ocurrió alguna.
        """
        self.shutdown()
