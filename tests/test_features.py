from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ml.features import (
    FBI_INDEX_CRIMES,
    FEATURE_COLS,
    build_features,
    build_single_record,
)


def _write_synthetic_parquet(path: Path) -> None:
    data = {
        "primary_type": ["THEFT", "HOMICIDE", "BATTERY", "ROBBERY", "THEFT"],
        "location_description": [
            "STREET",
            "RESIDENCE",
            "SIDEWALK",
            "STREET",
            "SIDEWALK",
        ],
        "domestic": [True, False, True, False, False],
        "year": [2020, 2021, 2022, 2023, 2024],
        "date": pd.to_datetime(
            [
                "2020-03-15 22:00:00",
                "2021-06-10 12:00:00",
                "2022-11-20 08:00:00",
                "2023-01-05 03:00:00",
                "2024-07-14 17:30:00",
            ]
        ),
        "hour": [22, 12, 8, 3, 17],
        "district": [1, 2, 3, 4, 5],
        "community_area": [8, 12, 25, 32, 44],
        "latitude": [41.88, 41.89, 41.87, 41.85, 41.83],
        "beat": [111, 222, 333, 444, 555],
        "longitude": [-87.63, -87.64, -87.62, -87.65, -87.61],
        "arrest": [True, False, True, False, True],
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    p = tmp_path / "crimes.parquet"
    _write_synthetic_parquet(p)
    return p


def test_build_features_columns(parquet_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    X, y, le_primary, le_location = build_features(str(parquet_path), conn)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y)
    assert len(X) == 5


def test_build_features_target_binary(parquet_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    X, y, _, _ = build_features(str(parquet_path), conn)
    assert set(y.unique()).issubset({0, 1})


def test_build_features_is_index_crime(parquet_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    X, _, _, _ = build_features(str(parquet_path), conn)
    # HOMICIDE and ROBBERY are index crimes; BATTERY is not
    assert X.loc[X.index[1], "is_index_crime"] == 1  # HOMICIDE
    assert X.loc[X.index[2], "is_index_crime"] == 0  # BATTERY


def test_build_features_reuses_encoders(parquet_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    _, _, le_primary, le_location = build_features(str(parquet_path), conn)
    # Second call reusing encoders should not fail
    conn2 = duckdb.connect(":memory:")
    X2, _, le_p2, le_l2 = build_features(
        str(parquet_path), conn2, le_primary, le_location
    )
    assert list(X2.columns) == FEATURE_COLS
    assert le_p2 is le_primary  # same object returned


def test_build_features_is_night(parquet_path: Path) -> None:
    conn = duckdb.connect(":memory:")
    X, _, _, _ = build_features(str(parquet_path), conn)
    # hour=22 → is_night=1; hour=12 → is_night=0
    assert X.iloc[0]["is_night"] == 1
    assert X.iloc[1]["is_night"] == 0


def test_build_single_record() -> None:
    import tempfile

    conn = duckdb.connect(":memory:")
    data = {
        "primary_type": ["THEFT", "BATTERY"],
        "location_description": ["STREET", "RESIDENCE"],
        "domestic": [False, True],
        "year": [2022, 2023],
        "date": pd.to_datetime(["2022-06-15 14:00:00", "2023-03-01 22:00:00"]),
        "hour": [14, 22],
        "district": [1, 2],
        "community_area": [8, 12],
        "latitude": [41.88, 41.89],
        "beat": [111, 222],
        "longitude": [-87.63, -87.64],
        "arrest": [True, False],
    }
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(data)), f.name)
        _, _, le_primary, le_location = build_features(f.name, conn)

    record = {
        "primary_type": "THEFT",
        "location_description": "STREET",
        "domestic": False,
        "year": 2024,
        "hour": 14,
        "district": 1,
        "community_area": 8,
        "latitude": 41.88,
        "beat": 111,
        "longitude": -87.63,
        "month": 6,
        "quarter": 2,
        "weekday": 2,
    }
    df = build_single_record(record, le_primary, le_location)
    assert list(df.columns) == FEATURE_COLS
    assert len(df) == 1
    assert df.iloc[0]["is_index_crime"] == 1  # THEFT is an FBI index crime
    assert df.iloc[0]["is_night"] == 0  # hour=14


def test_fbi_index_crimes_content() -> None:
    assert "HOMICIDE" in FBI_INDEX_CRIMES
    assert "THEFT" in FBI_INDEX_CRIMES
    assert "BATTERY" not in FBI_INDEX_CRIMES
