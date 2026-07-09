"""Tests para src/storage/duckdb_manager.py.

Estrategia: ningún test toca R2 ni MotherDuck reales.
- Validadores: tests puros sin mocks.
- Queries: motherduck_connection y r2_connection parcheados con MagicMock.
- load_settings: parcheado para devolver un MagicMock con los atributos necesarios.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow as pa
import pytest

from src.storage.duckdb_manager import (
    ALL_AREAS_SENTINEL,
    MAP_POINTS_LIMIT,
    YEAR_MAX,
    YEAR_MIN,
    DuckDBManager,
    _validate_limit,  # type: ignore[reportPrivateUsage]
    _validate_top_n,  # type: ignore[reportPrivateUsage]
    _validate_year,  # type: ignore[reportPrivateUsage]
    _validate_year_range,  # type: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings() -> MagicMock:
    cfg = MagicMock()
    cfg.r2_account_id.get_secret_value.return_value = "test-account"
    cfg.r2_access_key_id.get_secret_value.return_value = "test-key"
    cfg.r2_secret_access_key.get_secret_value.return_value = "test-secret"
    cfg.r2_bucket_name = "test-bucket"
    cfg.motherduck_database = "test-db"
    cfg.motherduck_token.get_secret_value.return_value = "test-md-token"
    return cfg


def _arrow(**columns: list[Any]) -> pa.Table:
    """Arrow table mínima para usar como return value en mocks."""
    return pa.table({k: pa.array(v) for k, v in columns.items()})


def _mock_conn(*, arrow: pa.Table | None = None, row: tuple[Any, ...] | None = None) -> MagicMock:
    """Crea un mock de DuckDBPyConnection preconfigurado."""
    conn = MagicMock()
    exec_ret = conn.execute.return_value
    if arrow is not None:
        exec_ret.arrow.return_value = arrow
        exec_ret.pl.return_value = pl.from_arrow(arrow)
    if row is not None:
        exec_ret.fetchone.return_value = row
    return conn


def _md_ctx(conn: MagicMock) -> MagicMock:
    """Context manager mock que devuelve conn al hacer __enter__."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_settings() -> Iterator[MagicMock]:
    """Parchea load_settings en todo el módulo duckdb_manager."""
    with patch(
        "src.storage.duckdb_manager.load_settings", return_value=_fake_settings()
    ) as mock:
        yield mock


@pytest.fixture()
def manager() -> DuckDBManager:
    return DuckDBManager()


# ---------------------------------------------------------------------------
# Validadores — tests puros, sin mocks
# ---------------------------------------------------------------------------


class TestValidateYear:
    def test_min_year_passes(self) -> None:
        _validate_year(YEAR_MIN, "year")

    def test_max_year_passes(self) -> None:
        _validate_year(YEAR_MAX, "year")

    def test_middle_year_passes(self) -> None:
        _validate_year(2015, "year")

    def test_below_min_raises(self) -> None:
        with pytest.raises(ValueError, match="year_start"):
            _validate_year(YEAR_MIN - 1, "year_start")

    def test_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="year_end"):
            _validate_year(YEAR_MAX + 1, "year_end")

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_year(2020.0, "year")  # type: ignore[arg-type]

    def test_string_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_year("2020", "year")  # type: ignore[arg-type]


class TestValidateYearRange:
    def test_valid_range(self) -> None:
        _validate_year_range(2010, 2020)

    def test_same_year_valid(self) -> None:
        _validate_year_range(2015, 2015)

    def test_start_greater_than_end_raises(self) -> None:
        with pytest.raises(ValueError, match="year_start.*mayor"):
            _validate_year_range(2020, 2010)

    def test_invalid_start_propagates(self) -> None:
        with pytest.raises(ValueError):
            _validate_year_range(1999, 2020)

    def test_invalid_end_propagates(self) -> None:
        with pytest.raises(ValueError):
            _validate_year_range(2010, 2030)


class TestValidateLimit:
    def test_min_limit(self) -> None:
        _validate_limit(1)

    def test_max_limit(self) -> None:
        _validate_limit(MAP_POINTS_LIMIT)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            _validate_limit(0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_limit(-1)

    def test_above_max_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_limit(MAP_POINTS_LIMIT + 1)

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_limit(100.0)  # type: ignore[arg-type]


class TestValidateTopN:
    def test_min(self) -> None:
        _validate_top_n(1)

    def test_max(self) -> None:
        _validate_top_n(50)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            _validate_top_n(0)

    def test_above_50_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_top_n(51)

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_top_n(10.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DuckDBManager — inicialización
# ---------------------------------------------------------------------------


class TestDuckDBManagerInit:
    def test_parquet_url_uses_bucket_name(self, manager: DuckDBManager) -> None:
        assert manager._parquet_url == "s3://test-bucket/crimes_full.parquet"  # type: ignore[reportPrivateUsage]

    def test_parquet_url_protocol(self, manager: DuckDBManager) -> None:
        assert manager._parquet_url.startswith("s3://")  # type: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# get_kpis
# ---------------------------------------------------------------------------


class TestGetKpis:
    def test_invalid_year_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_kpis(year_start=1990, year_end=2020)

    def test_returns_expected_keys(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(row=(1000, 200, 100, 20.0))
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            result = manager.get_kpis(2010, 2020)

        assert set(result) == {"total_crimes", "total_arrests", "total_domestic", "arrest_rate"}

    def test_null_row_values_default_to_zero(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(row=(None, None, None, None))
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            result = manager.get_kpis(2010, 2020)

        assert result["total_crimes"] == 0
        assert result["arrest_rate"] == 0.0

    def test_todas_sentinel_excludes_community_filter(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(row=(500, 50, 25, 10.0))
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_kpis(2010, 2020, community_area=ALL_AREAS_SENTINEL)

        sql, params = conn.execute.call_args.args
        assert "community_area" not in sql
        assert ALL_AREAS_SENTINEL not in params

    def test_community_area_filter_applied(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(row=(100, 10, 5, 10.0))
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_kpis(2010, 2020, community_area="LINCOLN PARK")

        _, params = conn.execute.call_args.args
        assert "LINCOLN PARK" in params

    def test_year_params_passed_to_query(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(row=(100, 10, 5, 10.0))
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_kpis(2015, 2022)

        _, params = conn.execute.call_args.args
        assert 2015 in params
        assert 2022 in params


# ---------------------------------------------------------------------------
# get_crimes_by_year
# ---------------------------------------------------------------------------


class TestGetCrimesByYear:
    _TABLE = _arrow(
        year=[2020, 2021],
        total_crimes=[100, 120],
        total_arrests=[10, 15],
        arrest_rate=[10.0, 12.5],
    )

    def test_invalid_year_range_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_crimes_by_year(2020, 2010)

    def test_returns_polars_dataframe(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            df = manager.get_crimes_by_year(2020, 2021)

        assert isinstance(df, pl.DataFrame)

    def test_todas_sentinel_excluded_from_filter(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_crimes_by_year(2020, 2021, community_area=ALL_AREAS_SENTINEL)

        sql, params = conn.execute.call_args.args
        assert "community_area" not in sql
        assert ALL_AREAS_SENTINEL not in params

    def test_community_area_added_to_params(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_crimes_by_year(2020, 2021, community_area="HYDE PARK")

        _, params = conn.execute.call_args.args
        assert "HYDE PARK" in params


# ---------------------------------------------------------------------------
# get_crimes_by_hour
# ---------------------------------------------------------------------------


class TestGetCrimesByHour:
    _TABLE = _arrow(
        hour=list(range(24)),
        total_crimes=[10] * 24,
        total_arrests=[1] * 24,
    )

    def test_invalid_year_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_crimes_by_hour(year_start=1999, year_end=2020)

    def test_year_params_passed_to_sql(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_crimes_by_hour(2018, 2022)

        _, params = conn.execute.call_args.args
        assert params == [2018, 2022]

    def test_returns_polars_dataframe(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            df = manager.get_crimes_by_hour(2018, 2022)

        assert isinstance(df, pl.DataFrame)


# ---------------------------------------------------------------------------
# get_crimes_by_type
# ---------------------------------------------------------------------------


class TestGetCrimesByType:
    _TABLE = _arrow(
        primary_type=["THEFT", "BATTERY"],
        total_crimes=[5000, 4000],
        total_arrests=[500, 400],
    )

    def test_invalid_top_n_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_crimes_by_type(top_n=0)

    def test_top_n_passed_to_query(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_crimes_by_type(top_n=5)

        _, params = conn.execute.call_args.args
        assert params == [5]


# ---------------------------------------------------------------------------
# get_heatmap_data
# ---------------------------------------------------------------------------


class TestGetHeatmapData:
    _TABLE = _arrow(
        community_area=["LOOP", "NEAR NORTH SIDE"],
        lat_centroid=[41.88, 41.90],
        lon_centroid=[-87.63, -87.64],
        total_crimes=[1000, 900],
    )

    def test_invalid_year_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_heatmap_data(year_start=2025, year_end=2020)

    def test_year_params_passed(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.motherduck_connection", return_value=_md_ctx(conn)
        ):
            manager.get_heatmap_data(2015, 2020)

        _, params = conn.execute.call_args.args
        assert params == [2015, 2020]


# ---------------------------------------------------------------------------
# get_map_points
# ---------------------------------------------------------------------------


class TestGetMapPoints:
    _TABLE = _arrow(
        latitude=[41.88],
        longitude=[-87.63],
        primary_type=["THEFT"],
        year=[2022],
        arrest=[False],
        community_area=["LOOP"],
    )

    def test_invalid_year_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_map_points(year_start=1990, year_end=2020)

    def test_invalid_limit_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.get_map_points(2010, 2020, limit=0)

    def test_todas_sentinel_excluded(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.r2_connection", return_value=_md_ctx(conn)
        ):
            manager.get_map_points(2010, 2020, community_area=ALL_AREAS_SENTINEL)

        sql, params = conn.execute.call_args.args
        # community_area aparece en SELECT pero no debe aparecer como filtro WHERE
        assert "community_area = ?" not in sql
        assert ALL_AREAS_SENTINEL not in params

    def test_community_area_filter_applied(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.r2_connection", return_value=_md_ctx(conn)
        ):
            manager.get_map_points(2010, 2020, community_area="AUSTIN")

        _, params = conn.execute.call_args.args
        assert "AUSTIN" in params

    def test_primary_type_filter_applied(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.r2_connection", return_value=_md_ctx(conn)
        ):
            manager.get_map_points(2010, 2020, primary_type="ROBBERY")

        _, params = conn.execute.call_args.args
        assert "ROBBERY" in params

    def test_parquet_url_first_param(self, manager: DuckDBManager) -> None:
        conn = _mock_conn(arrow=self._TABLE)
        with patch(
            "src.storage.duckdb_manager.r2_connection", return_value=_md_ctx(conn)
        ):
            manager.get_map_points(2010, 2020)

        _, params = conn.execute.call_args.args
        assert params[0] == "s3://test-bucket/crimes_full.parquet"


# ---------------------------------------------------------------------------
# build_agg_tables
# ---------------------------------------------------------------------------


class TestBuildAggTables:
    def test_calls_configure_s3(self, manager: DuckDBManager) -> None:
        conn = MagicMock()
        with (
            patch(
                "src.storage.duckdb_manager.motherduck_connection",
                return_value=_md_ctx(conn),
            ),
            patch("src.storage.duckdb_manager._configure_s3") as mock_cfg,
        ):
            manager.build_agg_tables()

        mock_cfg.assert_called_once_with(conn)

    def test_builds_all_five_tables(self, manager: DuckDBManager) -> None:
        conn = MagicMock()
        with patch(
            "src.storage.duckdb_manager.motherduck_connection",
            return_value=_md_ctx(conn),
        ):
            manager.build_agg_tables()

        sqls = [call_args.args[0] for call_args in conn.execute.call_args_list]
        table_names = {
            "agg_by_year",
            "agg_by_hour",
            "agg_by_type",
            "agg_domestic",
            "agg_heatmap",
        }
        for table in table_names:
            assert any(table in sql for sql in sqls), f"{table} no fue creada"


# ---------------------------------------------------------------------------
# refresh_year
# ---------------------------------------------------------------------------


class TestRefreshYear:
    def test_invalid_year_raises(self, manager: DuckDBManager) -> None:
        with pytest.raises(ValueError):
            manager.refresh_year(1999)

    def test_deletes_and_inserts_for_year(self, manager: DuckDBManager) -> None:
        conn = MagicMock()
        with patch(
            "src.storage.duckdb_manager.motherduck_connection",
            return_value=_md_ctx(conn),
        ):
            manager.refresh_year(2023)

        sqls = [call_args.args[0] for call_args in conn.execute.call_args_list]
        # Debe haber al menos un DELETE y un INSERT por cada tabla con año
        assert any("DELETE" in sql for sql in sqls)
        assert any("INSERT" in sql for sql in sqls)

    def test_year_param_in_delete_calls(self, manager: DuckDBManager) -> None:
        conn = MagicMock()
        with patch(
            "src.storage.duckdb_manager.motherduck_connection",
            return_value=_md_ctx(conn),
        ):
            manager.refresh_year(2023)

        delete_calls = [
            c for c in conn.execute.call_args_list if "DELETE" in c.args[0]
        ]
        for dc in delete_calls:
            assert dc.args[1] == [2023]

    def test_also_rebuilds_type_and_domestic(self, manager: DuckDBManager) -> None:
        conn = MagicMock()
        with patch(
            "src.storage.duckdb_manager.motherduck_connection",
            return_value=_md_ctx(conn),
        ):
            manager.refresh_year(2023)

        sqls = [c.args[0] for c in conn.execute.call_args_list]
        assert any("agg_by_type" in sql for sql in sqls)
        assert any("agg_domestic" in sql for sql in sqls)


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_both_up(self, manager: DuckDBManager) -> None:
        r2_conn = MagicMock()
        md_conn = MagicMock()
        with (
            patch(
                "src.storage.duckdb_manager.r2_connection",
                return_value=_md_ctx(r2_conn),
            ),
            patch(
                "src.storage.duckdb_manager.motherduck_connection",
                return_value=_md_ctx(md_conn),
            ),
        ):
            result = manager.health_check()

        assert result == {"r2": True, "motherduck": True}

    def test_r2_down(self, manager: DuckDBManager) -> None:
        failing_ctx = MagicMock()
        failing_ctx.__enter__ = MagicMock(side_effect=Exception("R2 connection failed"))
        failing_ctx.__exit__ = MagicMock(return_value=False)

        md_conn = MagicMock()
        with (
            patch(
                "src.storage.duckdb_manager.r2_connection", return_value=failing_ctx
            ),
            patch(
                "src.storage.duckdb_manager.motherduck_connection",
                return_value=_md_ctx(md_conn),
            ),
        ):
            result = manager.health_check()

        assert result["r2"] is False
        assert result["motherduck"] is True

    def test_motherduck_down(self, manager: DuckDBManager) -> None:
        r2_conn = MagicMock()
        failing_ctx = MagicMock()
        failing_ctx.__enter__ = MagicMock(
            side_effect=Exception("MotherDuck connection failed")
        )
        failing_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "src.storage.duckdb_manager.r2_connection",
                return_value=_md_ctx(r2_conn),
            ),
            patch(
                "src.storage.duckdb_manager.motherduck_connection",
                return_value=failing_ctx,
            ),
        ):
            result = manager.health_check()

        assert result["r2"] is True
        assert result["motherduck"] is False

    def test_both_down_returns_all_false(self, manager: DuckDBManager) -> None:
        failing_ctx = MagicMock()
        failing_ctx.__enter__ = MagicMock(side_effect=Exception("down"))
        failing_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "src.storage.duckdb_manager.r2_connection", return_value=failing_ctx
            ),
            patch(
                "src.storage.duckdb_manager.motherduck_connection",
                return_value=failing_ctx,
            ),
        ):
            result = manager.health_check()

        assert result == {"r2": False, "motherduck": False}
