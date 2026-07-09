"""Drift detection con Evidently — compara ventanas de 30 días."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import duckdb
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)

_DRIFT_COLUMNS = [
    "primary_type",
    "location_description",
    "domestic_int",
    "Year",
    "hour",
    "District",
    "Community Area",
    "Latitude",
    "Beat",
    "Longitude",
    "month",
    "quarter",
    "weekday",
    "is_index_crime",
    "is_night",
    "is_weekend",
]


def check_drift_from_dataframes(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.2,
) -> tuple[bool, dict[str, Any]]:
    """Runs Evidently DatasetDrift on two DataFrames.

    Args:
        reference: Baseline period (older window).
        current: Recent period to compare against reference.
        threshold: Fraction of drifted columns that triggers drift (default 0.2).

    Returns:
        Tuple of (drift_detected, report_dict) where report_dict always has:
        share_of_drifted_columns, number_of_drifted_columns,
        number_of_columns, dataset_drift.
    """
    cols = [
        c for c in _DRIFT_COLUMNS if c in reference.columns and c in current.columns
    ]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])
    result = report.as_dict()

    # Navigate Evidently result structure
    drift_result: dict[str, Any] = {}
    for metric in result.get("metrics", []):
        if "DatasetDriftMetric" in str(metric.get("metric", "")):
            drift_result = metric.get("result", {})
            break

    if not drift_result:
        logger.warning(
            "Evidently DatasetDriftMetric not found in report — "
            "check Evidently version compatibility. Treating as no drift."
        )

    share = float(drift_result.get("share_of_drifted_columns", 0.0))
    n_drifted = int(drift_result.get("number_of_drifted_columns", 0))
    n_total = int(drift_result.get("number_of_columns", len(cols)))
    drift_detected = share > threshold

    summary = {
        "share_of_drifted_columns": share,
        "number_of_drifted_columns": n_drifted,
        "number_of_columns": n_total,
        "dataset_drift": drift_detected,
    }
    logger.info(
        "Drift check: %.1f%% drifted columns (%d/%d) — detected=%s",
        share * 100,
        n_drifted,
        n_total,
        drift_detected,
    )
    return drift_detected, summary


def check_drift(
    parquet_url: str,
    conn: duckdb.DuckDBPyConnection,
    threshold: float = 0.2,
) -> tuple[bool, dict[str, Any]]:
    """Downloads last 60 days from R2 and runs drift check.

    Splits into reference (days 31–60 ago) and current (last 30 days).

    Args:
        parquet_url: S3 URL to crimes_full.parquet.
        conn: DuckDB connection with httpfs and R2 credentials configured.
        threshold: Fraction of drifted columns to trigger retraining.

    Returns:
        Tuple of (drift_detected, report_dict).
    """
    from src.ml.features import (
        FBI_INDEX_CRIMES,
        NIGHT_HOURS,
    )

    now = datetime.now(tz=UTC)
    cutoff_30 = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_60 = (now - timedelta(days=60)).strftime("%Y-%m-%d")

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
            EXTRACT(month FROM date)::INTEGER   AS month,
            EXTRACT(quarter FROM date)::INTEGER AS quarter,
            (ISODOW(date) - 1)::INTEGER         AS weekday,
            CAST(date AS DATE)                  AS day
        FROM read_parquet(?)
        WHERE date >= ?
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND primary_type IS NOT NULL
          AND location_description IS NOT NULL
        """,
        [parquet_url, cutoff_60],
    ).df()

    raw["is_index_crime"] = raw["primary_type"].isin(FBI_INDEX_CRIMES).astype(int)
    raw["domestic_int"] = raw["domestic"].astype(int)
    raw["is_night"] = raw["hour"].isin(NIGHT_HOURS).astype(int)
    raw["is_weekend"] = (raw["weekday"] >= 5).astype(int)
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

    reference = raw[raw["day"] < cutoff_30][_DRIFT_COLUMNS]
    current = raw[raw["day"] >= cutoff_30][_DRIFT_COLUMNS]

    if len(reference) < 100 or len(current) < 100:
        logger.warning(
            "Insufficient data for drift check (reference=%d, current=%d) — skipping",
            len(reference),
            len(current),
        )
        return False, {
            "share_of_drifted_columns": 0.0,
            "number_of_drifted_columns": 0,
            "number_of_columns": len(_DRIFT_COLUMNS),
            "dataset_drift": False,
        }

    return check_drift_from_dataframes(reference, current, threshold)
