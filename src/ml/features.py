"""Feature engineering compartido entre entrenamiento e inferencia."""

from __future__ import annotations

import duckdb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FBI_INDEX_CRIMES: frozenset[str] = frozenset(
    {
        "HOMICIDE",
        "CRIMINAL SEXUAL ASSAULT",
        "ROBBERY",
        "AGGRAVATED ASSAULT",
        "AGGRAVATED BATTERY",
        "BURGLARY",
        "THEFT",
        "MOTOR VEHICLE THEFT",
        "ARSON",
    }
)

NIGHT_HOURS: frozenset[int] = frozenset({22, 23, 0, 1, 2, 3, 4, 5})

FEATURE_COLS: list[str] = [
    "is_index_crime",
    "primary_type_enc",
    "location_enc",
    "domestic_int",
    "Year",
    "is_night",
    "hour",
    "District",
    "Community Area",
    "Latitude",
    "Beat",
    "Longitude",
    "month",
    "quarter",
    "weekday",
    "is_weekend",
]


def build_features(
    parquet_url: str,
    conn: duckdb.DuckDBPyConnection,
    le_primary: LabelEncoder | None = None,
    le_location: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder, LabelEncoder]:
    """Reads from parquet and builds X, y plus fitted LabelEncoders.

    If le_primary and le_location are provided (already fitted), they are
    reused directly without fitting. If None, they are fitted on this dataset.

    Args:
        parquet_url: Path or S3 URL to the crimes parquet file.
        conn: DuckDB connection (must have httpfs configured for S3 URLs).
        le_primary: Pre-fitted LabelEncoder for primary_type, or None.
        le_location: Pre-fitted LabelEncoder for location_description, or None.

    Returns:
        Tuple of (X, y, le_primary, le_location).
    """
    raw: pd.DataFrame = conn.execute(
        """
        SELECT
            primary_type,
            location_description,
            domestic,
            year,
            hour,
            district,
            community_area,
            latitude,
            beat,
            longitude,
            EXTRACT(month FROM date)::INTEGER    AS month,
            EXTRACT(quarter FROM date)::INTEGER  AS quarter,
            (ISODOW(date) - 1)::INTEGER          AS weekday,
            arrest
        FROM read_parquet(?)
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND primary_type IS NOT NULL
          AND location_description IS NOT NULL
        """,
        [parquet_url],
    ).df()

    raw["is_index_crime"] = raw["primary_type"].isin(FBI_INDEX_CRIMES).astype(int)
    raw["domestic_int"] = raw["domestic"].astype(int)
    raw["is_night"] = raw["hour"].isin(NIGHT_HOURS).astype(int)
    raw["is_weekend"] = (raw["weekday"] >= 5).astype(int)

    if le_primary is None:
        le_primary = LabelEncoder()
        le_primary.fit(raw["primary_type"])
    if le_location is None:
        le_location = LabelEncoder()
        le_location.fit(raw["location_description"])

    raw["primary_type_enc"] = le_primary.transform(raw["primary_type"])
    raw["location_enc"] = le_location.transform(raw["location_description"])

    raw = raw.rename(
        columns={
            "year": "Year",
            "district": "District",
            "community_area": "Community Area",
            "latitude": "Latitude",
            "beat": "Beat",
            "longitude": "Longitude",
        }
    )

    X = raw[FEATURE_COLS].copy()  # noqa: N806
    y = raw["arrest"].astype(int)
    return X, y, le_primary, le_location


def build_single_record(
    record: dict[str, object],
    le_primary: LabelEncoder,
    le_location: LabelEncoder,
) -> pd.DataFrame:
    """Transforms a single input dict into a 1-row feature DataFrame.

    Args:
        record: Dict with keys: primary_type, location_description, domestic,
            year, hour, district, community_area, latitude, beat, longitude,
            month, quarter, weekday.
        le_primary: Fitted LabelEncoder for primary_type.
        le_location: Fitted LabelEncoder for location_description.

    Returns:
        DataFrame with exactly the columns in FEATURE_COLS.
    """
    hour = int(record["hour"])  # type: ignore[call-overload]
    weekday = int(record["weekday"])  # type: ignore[call-overload]
    row = {
        "is_index_crime": int(str(record["primary_type"]) in FBI_INDEX_CRIMES),
        "primary_type_enc": int(le_primary.transform([record["primary_type"]])[0]),
        "location_enc": int(le_location.transform([record["location_description"]])[0]),
        "domestic_int": int(bool(record["domestic"])),
        "Year": record["year"],
        "is_night": int(hour in NIGHT_HOURS),
        "hour": hour,
        "District": record["district"],
        "Community Area": record["community_area"],
        "Latitude": record["latitude"],
        "Beat": record["beat"],
        "Longitude": record["longitude"],
        "month": record["month"],
        "quarter": record["quarter"],
        "weekday": weekday,
        "is_weekend": int(weekday >= 5),
    }
    return pd.DataFrame([row])[FEATURE_COLS]
