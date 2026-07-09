"""Carga inicial del dataset Chicago Crimes a R2 + MotherDuck.

Responsabilidades:
  - Recibir el archivo ya limpio desde Colab (crimes.db o crimes_full.parquet)
  - Exportar a crimes_full.parquet si viene como .db
  - Subir crimes_full.parquet a R2
  - Subir Parquets mensuales a R2 para el pipeline ML
  - Construir tablas agg_* en MotherDuck

Lo que NO hace:
  - Limpiar datos — eso ya lo hizo el notebook de Colab
  - Hacer JOIN con IUCR — ya está hecho en Colab
  - Feature engineering — ya está hecho en Colab

Uso:
    # Desde un .db de DuckDB/SQLite ya limpio
    uv run python -m src.ingestion.initial_load --source data/crimes.db

    # Desde un .parquet ya limpio (se salta la conversión)
    uv run python -m src.ingestion.initial_load --source data/crimes_full.parquet

    # Solo subir el Parquet, sin construir agg_* en MotherDuck
    uv run python -m src.ingestion.initial_load --source data/crimes.db --skip-aggs
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Literal

import polars as pl

from src.config import load_settings
from src.storage.duckdb_manager import DuckDBManager
from src.storage.r2 import R2BucketClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Nombre de la tabla dentro del .db de Colab
COLAB_TABLE_NAME = "crimes"

# Claves R2
R2_KEY_FULL = "crimes_full.parquet"
R2_KEY_MONTHLY = "crimes_by_month/{year}_{month:02d}.parquet"

# Compresión para todos los Parquets
PARQUET_COMPRESSION: Literal["zstd"] = "zstd"
PARQUET_COMPRESSION_LEVEL = 3


# ---------------------------------------------------------------------------
# Paso 1 — Leer el archivo ya limpio de Colab
# ---------------------------------------------------------------------------


def read_clean_file(source: Path) -> pl.DataFrame:
    """Lee el archivo ya limpio que viene de Colab.

    Soporta dos formatos:
      - .db  → DuckDB o SQLite exportado desde Colab
      - .parquet → Parquet exportado directamente desde Colab

    El archivo ya viene limpio — no se aplica ninguna transformación.

    Args:
        source: Ruta al archivo .db o .parquet local.

    Returns:
        DataFrame Polars con los datos limpios de Colab.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado.
    """
    if not source.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {source}")

    suffix = source.suffix.lower()
    logger.info("Leyendo archivo limpio de Colab: %s", source)
    start = time.perf_counter()

    if suffix == ".parquet":
        df = pl.read_parquet(str(source))

    elif suffix == ".db":
        # DuckDB/SQLite exportado desde Colab
        df = pl.read_database_uri(
            query=f"SELECT * FROM {COLAB_TABLE_NAME}",
            uri=f"sqlite:///{source}",
        )

    else:
        raise ValueError(
            f"Formato no soportado: '{suffix}'. "
            "Use .parquet o .db (DuckDB/SQLite exportado desde Colab)."
        )

    elapsed = time.perf_counter() - start
    logger.info(
        "Lectura completada: %d filas, %d columnas en %.1fs",
        len(df),
        len(df.columns),
        elapsed,
    )

    return df


# ---------------------------------------------------------------------------
# Paso 2 — Normalización de nombres y tipos del Parquet
# ---------------------------------------------------------------------------


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza nombres y tipos del Parquet exportado por Colab.

    El CSV de Chicago tiene nombres con mayúsculas y espacios.
    Esta función los convierte al snake_case que usa el pipeline,
    parsea Date a Datetime, extrae hour, y corrige coordenadas
    con coma decimal.

    Args:
        df: DataFrame tal como viene del Parquet de Colab.

    Returns:
        DataFrame con columnas normalizadas listo para R2.
    """
    logger.info("Normalizando columnas del Parquet de Colab...")

    return (
        df
        # Paso 1 — Parsear Date a Datetime real
        .with_columns(
            pl.col("Date")
            .str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False)
            .alias("date")
        )
        # Paso 2 — Extraer hour desde el Datetime ya parseado
        .with_columns(pl.col("date").dt.hour().alias("hour"))
        # Paso 3 — Corregir coordenadas con coma decimal → punto
        .with_columns(
            [
                pl.col("Latitude")
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias("Latitude"),
                pl.col("Longitude")
                .str.replace(",", ".")
                .cast(pl.Float64, strict=False)
                .alias("Longitude"),
            ]
        )
        # Paso 4 — Renombrar a snake_case y eliminar Date original
        .drop("Date")
        .rename(
            {
                "Year": "year",
                "Primary Type": "primary_type",
                "Description": "description",
                "Location Description": "location_description",
                "Arrest": "arrest",
                "Domestic": "domestic",
                "Beat": "beat",
                "District": "district",
                "Ward": "ward",
                "Community Area": "community_area",
                "Latitude": "latitude",
                "Longitude": "longitude",
            }
        )
    )


# ---------------------------------------------------------------------------
# Paso 3 — Exportar a Parquet completo y subir a R2
# ---------------------------------------------------------------------------


def export_and_upload_full(df: pl.DataFrame, tmp_path: Path) -> None:
    """Exporta el DataFrame a Parquet y lo sube a R2.

    Args:
        df:       DataFrame limpio de Colab.
        tmp_path: Directorio temporal para el archivo Parquet.
    """
    parquet_path = tmp_path / "crimes_full.parquet"

    # Exportar a Parquet local
    logger.info("Exportando crimes_full.parquet (%d filas)...", len(df))
    start = time.perf_counter()

    df.write_parquet(
        str(parquet_path),
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        statistics=True,  # mejora el lazy loading de DuckDB
    )

    size_mb = parquet_path.stat().st_size / (1024**2)
    logger.info(
        "Parquet generado: %.1f MB en %.1fs",
        size_mb,
        time.perf_counter() - start,
    )

    # Subir a R2
    cfg = load_settings()
    logger.info("Subiendo a R2 como '%s'...", R2_KEY_FULL)
    start = time.perf_counter()

    with R2BucketClient(cfg) as client:
        client.upload_file(
            local_path=parquet_path,
            key=R2_KEY_FULL,
            content_type="application/octet-stream",
        )

    logger.info("Upload completado en %.1fs", time.perf_counter() - start)
    parquet_path.unlink()


# ---------------------------------------------------------------------------
# Paso 3 — Exportar Parquets mensuales a R2 (para pipeline ML)
# ---------------------------------------------------------------------------


def export_and_upload_monthly(df: pl.DataFrame, tmp_path: Path) -> None:
    """Exporta y sube un Parquet por mes a R2 para reentrenamiento de XGBoost.

    Estructura en R2:
        crimes_by_month/2023_01.parquet
        crimes_by_month/2023_02.parquet
        ...

    Requiere columnas 'year' y 'date' en el DataFrame.

    Args:
        df:       DataFrame limpio de Colab.
        tmp_path: Directorio temporal para archivos intermedios.
    """
    if "year" not in df.columns or "date" not in df.columns:
        logger.warning(
            "Columnas 'year' o 'date' no encontradas — saltando Parquets mensuales"
        )
        return

    # Extraer mes desde la columna date
    df_with_month = df.with_columns(pl.col("date").dt.month().alias("month"))

    year_months = (
        df_with_month.select(["year", "month"]).unique().sort(["year", "month"]).rows()
    )

    logger.info("Exportando %d archivos mensuales a R2...", len(year_months))
    cfg = load_settings()

    with R2BucketClient(cfg) as client:
        for year, month in year_months:
            monthly_df = df_with_month.filter(
                (pl.col("year") == year) & (pl.col("month") == month)
            ).drop("month")  # no necesitamos la columna extra en el Parquet

            parquet_path = tmp_path / f"{year}_{month:02d}.parquet"
            monthly_df.write_parquet(
                str(parquet_path),
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                statistics=True,
            )

            r2_key = R2_KEY_MONTHLY.format(year=year, month=month)
            client.upload_file(
                local_path=parquet_path,
                key=r2_key,
                content_type="application/octet-stream",
            )

            # Limpiar inmediatamente para no acumular en disco
            parquet_path.unlink()
            logger.info("  ✓ %s (%d filas)", r2_key, len(monthly_df))

    logger.info("Parquets mensuales completados.")


# ---------------------------------------------------------------------------
# Paso 4 — Construir tablas agg_* en MotherDuck
# ---------------------------------------------------------------------------


def build_agg_tables() -> None:
    """Construye todas las tablas agg_* en MotherDuck desde R2.

    Lee crimes_full.parquet desde R2 via httpfs —
    no necesita el archivo local en este paso.
    """
    logger.info("Construyendo tablas agg_* en MotherDuck...")
    start = time.perf_counter()

    DuckDBManager().build_agg_tables()

    logger.info(
        "Tablas agg_* completadas en %.1fs",
        time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Paso 5 — Verificación final
# ---------------------------------------------------------------------------


def verify_load() -> None:
    """Verifica que R2 y MotherDuck estén accesibles tras la carga.

    Raises:
        RuntimeError: Si algún servicio no responde.
    """
    logger.info("Verificando acceso a R2 y MotherDuck...")

    status = DuckDBManager().health_check()

    if not status["r2"]:
        raise RuntimeError("R2 no responde — verificar credenciales y bucket")
    if not status["motherduck"]:
        raise RuntimeError("MotherDuck no responde — verificar token")

    logger.info(
        "Verificación OK: R2=%s, MotherDuck=%s",
        status["r2"],
        status["motherduck"],
    )


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------


def run_initial_load(source: Path, skip_aggs: bool = False) -> None:
    """Orquesta la carga inicial completa.

    Pasos:
        1. Leer el archivo ya limpio de Colab (.db o .parquet)
        2. Subir crimes_full.parquet a R2
        3. Subir Parquets mensuales a R2
        4. Construir tablas agg_* en MotherDuck (opcional)
        5. Verificar acceso final

    Args:
        source:     Ruta al archivo limpio de Colab.
        skip_aggs:  Si True, salta la construcción de agg_*.
    """
    total_start = time.perf_counter()
    logger.info("=== Iniciando carga inicial Chicago Crimes ===")
    logger.info("Fuente (ya limpia por Colab): %s", source)
    logger.info("skip_aggs: %s", skip_aggs)

    tmp_path = Path("data") / "tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        # Paso 1 — Leer archivo limpio de Colab
        df = read_clean_file(source)

        # Paso 1.5 — Normalizar columnas
        df = normalize_columns(df)

        # Paso 2 — Subir Parquet completo a R2
        export_and_upload_full(df, tmp_path)

        # Paso 3 — Subir Parquets mensuales a R2
        export_and_upload_monthly(df, tmp_path)

        # Paso 4 — Construir agg_* en MotherDuck
        if not skip_aggs:
            build_agg_tables()
        else:
            logger.info("skip_aggs=True — saltando construcción de agg_*")

        # Paso 5 — Verificar
        verify_load()

    finally:
        # Limpiar temporales aunque haya errores
        if tmp_path.exists():
            for f in tmp_path.iterdir():
                f.unlink(missing_ok=True)
            tmp_path.rmdir()

    logger.info(
        "=== Carga inicial completada en %.1fs ===",
        time.perf_counter() - total_start,
    )


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carga inicial Chicago Crimes a R2 + MotherDuck. "
            "El archivo fuente debe venir ya limpio desde el notebook de Colab."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Ruta al archivo limpio de Colab. "
            "Acepta .parquet o .db (DuckDB/SQLite). "
            "Ejemplo: data/crimes.db"
        ),
    )
    parser.add_argument(
        "--skip-aggs",
        action="store_true",
        default=False,
        help="Salta la construcción de tablas agg_* en MotherDuck",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    run_initial_load(source=args.source, skip_aggs=args.skip_aggs)
