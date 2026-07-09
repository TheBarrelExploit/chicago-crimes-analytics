"""Gestor de conexiones DuckDB para Chicago Crimes Analytics.

Responsabilidades:
  - Configurar DuckDB con httpfs para leer Parquet desde R2
  - Conectar a MotherDuck para leer/escribir tablas agg_*
  - Proveer queries reutilizables para el dashboard y el ETL
  - Nunca cargar los 8.5M filas completos en memoria
  - Queries sanitizadas con parámetros — sin f-strings con user input
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
import polars as pl

from src.config import load_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Rango de años válido del dataset
YEAR_MIN = 2001
YEAR_MAX = 2026

# Máximo de puntos para el mapa (evita saturar el browser)
MAP_POINTS_LIMIT = 5_000

# Top N tipos de crimen por defecto
DEFAULT_TOP_N = 12

# Sentinel para indicar "todas las áreas" en los filtros del dashboard
ALL_AREAS_SENTINEL = "todas"


# ---------------------------------------------------------------------------
# Validadores internos
# ---------------------------------------------------------------------------


def _validate_year(year: int, name: str) -> None:
    """Valida que el año esté dentro del rango permitido del dataset."""
    if type(year) is not int or not (YEAR_MIN <= year <= YEAR_MAX):
        raise ValueError(
            f"{name} debe ser un entero entre {YEAR_MIN} y {YEAR_MAX}, "
            f"recibido: {year!r}"
        )


def _validate_year_range(year_start: int, year_end: int) -> None:
    """Valida el rango de años completo."""
    _validate_year(year_start, "year_start")
    _validate_year(year_end, "year_end")
    if year_start > year_end:
        raise ValueError(
            f"year_start ({year_start}) no puede ser mayor que year_end ({year_end})"
        )


def _validate_limit(limit: int) -> None:
    """Valida que el límite de filas sea positivo y razonable."""
    if type(limit) is not int or not (1 <= limit <= MAP_POINTS_LIMIT):
        raise ValueError(
            f"limit debe ser un entero entre 1 y {MAP_POINTS_LIMIT}, "
            f"recibido: {limit!r}"
        )


def _validate_top_n(top_n: int) -> None:
    """Valida que top_n sea un entero positivo razonable."""
    if type(top_n) is not int or not (1 <= top_n <= 50):
        raise ValueError(f"top_n debe ser un entero entre 1 y 50, recibido: {top_n!r}")


# ---------------------------------------------------------------------------
# Factory de conexiones
# ---------------------------------------------------------------------------


def _configure_s3(conn: duckdb.DuckDBPyConnection) -> None:
    """Instala httpfs y configura credenciales S3/R2 en una conexión existente."""
    cfg = load_settings()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(
        "SET s3_endpoint=?",
        [f"{cfg.r2_account_id.get_secret_value()}.r2.cloudflarestorage.com"],
    )
    conn.execute("SET s3_access_key_id=?", [cfg.r2_access_key_id.get_secret_value()])
    conn.execute(
        "SET s3_secret_access_key=?",
        [cfg.r2_secret_access_key.get_secret_value()],
    )
    conn.execute("SET s3_region='auto';")
    conn.execute("SET s3_url_style='path';")


def _make_r2_connection() -> duckdb.DuckDBPyConnection:
    """Crea una conexión DuckDB configurada para leer Parquet desde R2.

    Usa httpfs — nunca carga el archivo completo en memoria (lazy loading).
    Las credenciales vienen de config, no de user input — seguras con SET.
    """
    cfg = load_settings()
    conn = duckdb.connect(database=":memory:")
    _configure_s3(conn)
    logger.info("Conexión R2 configurada para bucket: %s", cfg.r2_bucket_name)
    return conn


def _make_motherduck_connection() -> duckdb.DuckDBPyConnection:
    """Crea una conexión a MotherDuck para leer/escribir tablas agg_*.

    Requiere MOTHERDUCK_TOKEN en las variables de entorno.
    """
    cfg = load_settings()
    if cfg.motherduck_token is None:
        raise ValueError("MOTHERDUCK_TOKEN no está configurado en el entorno")
    conn = duckdb.connect(
        database=f"md:{cfg.motherduck_database}",
        config={"motherduck_token": cfg.motherduck_token.get_secret_value()},
    )

    logger.info("Conexión MotherDuck establecida: %s", cfg.motherduck_database)
    return conn


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def r2_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager para queries a R2 via httpfs."""
    conn = _make_r2_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def motherduck_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager para queries a MotherDuck (tablas agg_*)."""
    conn = _make_motherduck_connection()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DuckDBManager
# ---------------------------------------------------------------------------


class DuckDBManager:
    """Clase principal para interactuar con DuckDB en el proyecto.

    Dos modos de operación:
      - R2: queries lazy sobre crimes_full.parquet (sin cargar en memoria)
      - MotherDuck: queries rápidas sobre tablas agg_* pre-calculadas

    Seguridad:
      - Todo user input pasa como parámetro ? — nunca interpolado en SQL
      - Valores numéricos validados con rangos antes de la query
      - Strings de configuración interna nunca expuestos al usuario

    Uso típico en el dashboard:
        manager = DuckDBManager()
        df = manager.get_crimes_by_year(year_start=2010, year_end=2020)
    """

    def __init__(self) -> None:
        """Inicializa la instancia de la clase.

        Carga las configuraciones globales del sistema y construye
        la URL de conexión para el archivo Parquet alojado en S3/R2.

        Attributes:
            _cfg (Config): Objeto que contiene la configuración cargada (settings).
            _parquet_url (str): Ruta completa al archivo 'crimes_full.parquet' en el bucket.
        """
        self._cfg = load_settings()
        self._parquet_url = f"s3://{self._cfg.r2_bucket_name}/crimes_full.parquet"

    # ------------------------------------------------------------------
    # Queries para el dashboard — van a MotherDuck (agg_*)
    # ------------------------------------------------------------------

    def get_kpis(
        self,
        year_start: int = 2001,
        year_end: int = 2026,
        community_area: str | None = None,
        primary_type: str | None = None,
        domestic: bool | None = None,
    ) -> dict[str, float]:
        """Retorna los KPIs principales del dashboard.

        Calcula la tasa de arresto al vuelo como hacía DAX.

        Args:
            year_start: Año de inicio del filtro.
            year_end: Año de fin del filtro.
            community_area: Código numérico del área como string. None = todas.
            primary_type: Tipo de crimen exacto. None = todos.
            domestic: True = solo domésticos, False = solo no domésticos, None = todos.

        Returns:
            Diccionario con total_crimes, total_arrests, total_domestic y arrest_rate (%).
        """
        _validate_year_range(year_start, year_end)

        conditions = ["year BETWEEN ? AND ?"]
        params: list[int | str | bool] = [year_start, year_end]

        if community_area and community_area.lower() != ALL_AREAS_SENTINEL:
            conditions.append("community_area = ?")
            params.append(community_area)

        if primary_type:
            conditions.append("primary_type = ?")
            params.append(primary_type)

        if domestic is not None:
            conditions.append("domestic = ?")
            params.append(domestic)

        where = "WHERE " + " AND ".join(conditions)

        with motherduck_connection() as conn:
            row = conn.execute(
                f"""
                SELECT
                    SUM(total_crimes)                                              AS total_crimes,
                    SUM(total_arrests)                                             AS total_arrests,
                    SUM(CASE WHEN domestic THEN total_crimes ELSE 0 END)           AS total_domestic,
                    ROUND(
                        SUM(total_arrests) * 100.0 / NULLIF(SUM(total_crimes), 0),
                        2
                    )                                                              AS arrest_rate
                FROM agg_by_year
                {where}
                """,
                params,
            ).fetchone()

            if row is None:
                return {
                    "total_crimes": 0,
                    "total_arrests": 0,
                    "total_domestic": 0,
                    "arrest_rate": 0.0,
                }

            return {
                "total_crimes": int(row[0] or 0),
                "total_arrests": int(row[1] or 0),
                "total_domestic": int(row[2] or 0),
                "arrest_rate": float(row[3] or 0.0),
            }

    def get_crimes_by_year(
        self,
        year_start: int = 2001,
        year_end: int = 2026,
        community_area: str | None = None,
        primary_type: str | None = None,
        domestic: bool | None = None,
    ) -> pl.DataFrame:
        """Gráfico de línea doble eje: crímenes + tasa arresto por año.

        Calcula arrest_rate al vuelo como DAX.

        Args:
            year_start: Año de inicio del filtro.
            year_end: Año de fin del filtro.
            community_area: Código numérico del área como string. None = todas.
            primary_type: Tipo de crimen exacto. None = todos.
            domestic: True = solo domésticos, False = solo no domésticos, None = todos.
        """
        _validate_year_range(year_start, year_end)

        conditions = ["year BETWEEN ? AND ?"]
        params: list[int | str | bool] = [year_start, year_end]

        if community_area and community_area.lower() != ALL_AREAS_SENTINEL:
            conditions.append("community_area = ?")
            params.append(community_area)

        if primary_type:
            conditions.append("primary_type = ?")
            params.append(primary_type)

        if domestic is not None:
            conditions.append("domestic = ?")
            params.append(domestic)

        where = "WHERE " + " AND ".join(conditions)

        with motherduck_connection() as conn:
            return conn.execute(
                f"""
                SELECT
                    year,
                    SUM(total_crimes)                                    AS total_crimes,
                    SUM(total_arrests)                                   AS total_arrests,
                    SUM(CASE WHEN domestic THEN total_crimes ELSE 0 END) AS total_domestic,
                    ROUND(
                        SUM(total_arrests) * 100.0 / NULLIF(SUM(total_crimes), 0),
                        2
                    )                                                    AS arrest_rate
                FROM agg_by_year
                {where}
                GROUP BY year
                ORDER BY year
                """,
                params,
            ).pl()

    def get_crimes_by_hour(
        self,
        year_start: int = 2001,
        year_end: int = 2026,
    ) -> pl.DataFrame:
        """Gráfico de barras: crímenes por hora del día (0-23)."""
        _validate_year_range(year_start, year_end)

        with motherduck_connection() as conn:
            return conn.execute(
                """
                SELECT
                    hour,
                    SUM(total_crimes)  AS total_crimes,
                    SUM(total_arrests) AS total_arrests
                FROM agg_by_hour
                WHERE year BETWEEN ? AND ?
                GROUP BY hour
                ORDER BY hour
                """,
                [year_start, year_end],
            ).pl()

    def get_crimes_by_type(
        self,
        top_n: int = DEFAULT_TOP_N,
    ) -> pl.DataFrame:
        """Barras horizontales: top N tipos de crimen."""
        _validate_top_n(top_n)

        with motherduck_connection() as conn:
            return conn.execute(
                """
                SELECT
                    primary_type,
                    SUM(total_crimes)  AS total_crimes,
                    SUM(total_arrests) AS total_arrests
                FROM agg_by_type
                GROUP BY primary_type
                ORDER BY total_crimes DESC
                LIMIT ?
                """,
                [top_n],
            ).pl()

    def get_domestic_split(self) -> pl.DataFrame:
        """Pie chart: doméstico vs no doméstico."""
        with motherduck_connection() as conn:
            return conn.execute(
                """
                SELECT
                    domestic,
                    SUM(total_crimes) AS total_crimes,
                    ROUND(
                        SUM(total_crimes) * 100.0 / SUM(SUM(total_crimes)) OVER (),
                        2
                    ) AS percentage
                FROM agg_domestic
                GROUP BY domestic
                """,
            ).pl()

    def get_heatmap_data(
        self,
        year_start: int = 2001,
        year_end: int = 2026,
    ) -> pl.DataFrame:
        """Mapa de burbujas por community area.

        Tamaño de burbuja = total_crimes agregado del rango de años.S
        """
        _validate_year_range(year_start, year_end)

        with motherduck_connection() as conn:
            return conn.execute(
                """
                SELECT
                    community_area,
                    AVG(lat_centroid) AS lat_centroid,
                    AVG(lon_centroid) AS lon_centroid,
                    SUM(total_crimes) AS total_crimes
                FROM agg_heatmap
                WHERE year BETWEEN ? AND ?
                GROUP BY community_area
                ORDER BY total_crimes DESC
                """,
                [year_start, year_end],
            ).pl()

    # ------------------------------------------------------------------
    # Queries sobre R2 — lazy loading para casos específicos
    # ------------------------------------------------------------------

    def get_map_points(
        self,
        year_start: int,
        year_end: int,
        community_area: str | None = None,
        primary_type: str | None = None,
        limit: int = MAP_POINTS_LIMIT,
    ) -> pl.DataFrame:
        """Puntos individuales para el mapa con filtros específicos.

        Lee desde R2 con lazy loading — nunca carga el Parquet completo.

        Seguridad:
          - year_start / year_end: validados como enteros en rango
          - community_area / primary_type: pasados como parámetros ?
          - limit: validado con _validate_limit
        """
        _validate_year_range(year_start, year_end)
        _validate_limit(limit)

        conditions = [
            "year BETWEEN ? AND ?",
            "latitude IS NOT NULL",
            "longitude IS NOT NULL",
        ]
        params: list[int | str] = [year_start, year_end]

        if community_area and community_area.lower() != ALL_AREAS_SENTINEL:
            conditions.append("community_area = ?")
            params.append(community_area)

        if primary_type:
            conditions.append("primary_type = ?")
            params.append(primary_type)

        where = "WHERE " + " AND ".join(conditions)

        # USING SAMPLE evita el ORDER BY RANDOM() costoso
        with r2_connection() as conn:
            result = conn.execute(
                f"""
                SELECT
                    latitude,
                    longitude,
                    primary_type,
                    year,
                    arrest,
                    community_area
                FROM read_parquet(?)
                {where}
                USING SAMPLE ?
                """,
                [self._parquet_url] + params + [limit],
            ).pl()

        logger.info(
            "Map points query: %d-%d, area=%s, type=%s, limit=%d",
            year_start,
            year_end,
            community_area,
            primary_type,
            limit,
        )
        return result

    # ------------------------------------------------------------------
    # ETL — construir tablas agg_* en MotherDuck desde R2
    # ------------------------------------------------------------------

    def build_agg_tables(self) -> None:
        """Calcula y guarda todas las tablas agg_* en MotherDuck.

        Se ejecuta una vez en el ETL, no en el dashboard.

        Nota: las queries ETL usan la URL del Parquet como parámetro
        interno — no viene de user input, por eso es seguro.
        """
        logger.info("Iniciando construcción de tablas agg_* en MotherDuck...")

        with motherduck_connection() as conn:
            _configure_s3(conn)
            self._build_agg_by_year(conn)
            self._build_agg_by_hour(conn)
            self._build_agg_by_type(conn)
            self._build_agg_domestic(conn)
            self._build_agg_heatmap(conn)

        logger.info("Tablas agg_* construidas exitosamente en MotherDuck.")

    def refresh_year(self, year: int) -> None:
        """Actualiza los datos de un año específico en las tablas agg_*.

        Más eficiente que build_agg_tables() cuando solo cambian datos de un año
        (p.ej. al incorporar el año en curso o corregir datos recientes).

        Las tablas sin columna year (agg_by_type, agg_domestic) se reconstruyen
        completamente porque agregan sobre todo el historial.

        Args:
            year(int): Año a refrescar. Debe estar dentro de [YEAR_MIN, YEAR_MAX].
        """
        _validate_year(year, "year")
        logger.info("Actualizando tablas agg_* para el año %d...", year)

        with motherduck_connection() as conn:
            _configure_s3(conn)
            self._refresh_agg_by_year(conn, year)
            self._refresh_agg_by_hour(conn, year)
            self._refresh_agg_heatmap(conn, year)
            self._build_agg_by_type(conn)
            self._build_agg_domestic(conn)

        logger.info("Tablas agg_* actualizadas para el año %d.", year)

    def _refresh_agg_by_year(self, conn: duckdb.DuckDBPyConnection, year: int) -> None:
        logger.info("Refrescando agg_by_year para %d...", year)
        conn.execute("DELETE FROM agg_by_year WHERE year = ?", [year])
        conn.execute(
            """
            INSERT INTO agg_by_year
            SELECT
                year,
                community_area,
                primary_type,
                domestic,
                COUNT(*)         AS total_crimes,
                SUM(arrest::INT) AS total_arrests
            FROM read_parquet(?)
            WHERE year = ?
            GROUP BY year, community_area, primary_type, domestic
            """,
            [self._parquet_url, year],
        )

    def _refresh_agg_by_hour(self, conn: duckdb.DuckDBPyConnection, year: int) -> None:
        logger.info("Refrescando agg_by_hour para %d...", year)
        conn.execute("DELETE FROM agg_by_hour WHERE year = ?", [year])
        conn.execute(
            """
            INSERT INTO agg_by_hour
            SELECT
                year,
                hour,
                COUNT(*)         AS total_crimes,
                SUM(arrest::INT) AS total_arrests
            FROM read_parquet(?)
            WHERE year = ?
            GROUP BY year, hour
            ORDER BY year, hour
            """,
            [self._parquet_url, year],
        )

    def _refresh_agg_heatmap(self, conn: duckdb.DuckDBPyConnection, year: int) -> None:
        logger.info("Refrescando agg_heatmap para %d...", year)
        conn.execute("DELETE FROM agg_heatmap WHERE year = ?", [year])
        conn.execute(
            """
            INSERT INTO agg_heatmap
            SELECT
                community_area,
                year,
                AVG(latitude)  AS lat_centroid,
                AVG(longitude) AS lon_centroid,
                COUNT(*)       AS total_crimes
            FROM read_parquet(?)
            WHERE year = ?
              AND latitude  IS NOT NULL
              AND longitude IS NOT NULL
            GROUP BY community_area, year
            ORDER BY community_area, year
            """,
            [self._parquet_url, year],
        )

    def _build_agg_by_year(self, conn: duckdb.DuckDBPyConnection) -> None:
        logger.info("Construyendo agg_by_year...")
        conn.execute(
            """
            CREATE OR REPLACE TABLE agg_by_year AS
            SELECT
                year,
                community_area,
                primary_type,
                domestic,
                COUNT(*)         AS total_crimes,
                SUM(arrest::INT) AS total_arrests
            FROM read_parquet(?)
            GROUP BY year, community_area, primary_type, domestic
            ORDER BY year, community_area, primary_type, domestic
            """,
            [self._parquet_url],
        )

    def _build_agg_by_hour(self, conn: duckdb.DuckDBPyConnection) -> None:
        logger.info("Construyendo agg_by_hour...")
        conn.execute(
            """
            CREATE OR REPLACE TABLE agg_by_hour AS
            SELECT
                year,
                hour,
                COUNT(*)         AS total_crimes,
                SUM(arrest::INT) AS total_arrests
            FROM read_parquet(?)
            GROUP BY year, hour
            ORDER BY year, hour
            """,
            [self._parquet_url],
        )

    def _build_agg_by_type(self, conn: duckdb.DuckDBPyConnection) -> None:
        logger.info("Construyendo agg_by_type...")
        conn.execute(
            """
            CREATE OR REPLACE TABLE agg_by_type AS
            SELECT
                primary_type,
                COUNT(*)         AS total_crimes,
                SUM(arrest::INT) AS total_arrests
            FROM read_parquet(?)
            GROUP BY primary_type
            ORDER BY total_crimes DESC
            """,
            [self._parquet_url],
        )

    def _build_agg_domestic(self, conn: duckdb.DuckDBPyConnection) -> None:
        logger.info("Construyendo agg_domestic...")
        conn.execute(
            """
            CREATE OR REPLACE TABLE agg_domestic AS
            SELECT
                domestic,
                COUNT(*) AS total_crimes
            FROM read_parquet(?)
            GROUP BY domestic
            """,
            [self._parquet_url],
        )

    def _build_agg_heatmap(self, conn: duckdb.DuckDBPyConnection) -> None:
        logger.info("Construyendo agg_heatmap...")
        conn.execute(
            """
            CREATE OR REPLACE TABLE agg_heatmap AS
            SELECT
                community_area,
                year,
                AVG(latitude)  AS lat_centroid,
                AVG(longitude) AS lon_centroid,
                COUNT(*)       AS total_crimes
            FROM read_parquet(?)
            WHERE latitude  IS NOT NULL
              AND longitude IS NOT NULL
            GROUP BY community_area, year
            ORDER BY community_area, year
            """,
            [self._parquet_url],
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, bool]:
        """Verifica que R2 y MotherDuck estén accesibles.

        Útil para el startup de FastAPI y Streamlit.
        """
        status = {"r2": False, "motherduck": False}

        try:
            with r2_connection() as conn:
                conn.execute(
                    "SELECT COUNT(*) FROM read_parquet(?) LIMIT 1", [self._parquet_url]
                )
            status["r2"] = True
        except Exception as e:
            logger.error("R2 health check falló: %s", e)

        try:
            with motherduck_connection() as conn:
                conn.execute("SELECT 1")
            status["motherduck"] = True
        except Exception as e:
            logger.error("MotherDuck health check falló: %s", e)

        return status
