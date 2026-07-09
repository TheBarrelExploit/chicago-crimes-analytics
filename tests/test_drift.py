from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.drift import check_drift_from_dataframes

_CRIME_TYPES = ["THEFT", "BATTERY", "ROBBERY", "ASSAULT", "BURGLARY"]
_LOCATIONS = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "PARKING LOT"]


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "primary_type": rng.choice(_CRIME_TYPES, n),
            "location_description": rng.choice(_LOCATIONS, n),
            "domestic_int": rng.integers(0, 2, n),
            "Year": rng.integers(2018, 2024, n),
            "hour": rng.integers(0, 24, n),
            "District": rng.integers(1, 25, n),
            "Community Area": rng.integers(1, 77, n),
            "Latitude": rng.uniform(41.6, 42.0, n),
            "Beat": rng.integers(100, 2500, n),
            "Longitude": rng.uniform(-87.9, -87.5, n),
            "month": rng.integers(1, 13, n),
            "quarter": rng.integers(1, 5, n),
            "weekday": rng.integers(0, 7, n),
            "is_index_crime": rng.integers(0, 2, n),
            "is_night": rng.integers(0, 2, n),
            "is_weekend": rng.integers(0, 2, n),
        }
    )


def test_no_drift_identical_distributions() -> None:
    reference = _make_df(500, seed=1)
    current = _make_df(500, seed=2)  # same distribution, different seed
    detected, report = check_drift_from_dataframes(reference, current)
    assert isinstance(detected, bool)
    assert "share_of_drifted_columns" in report
    # Identical distribution → should not trigger drift
    assert not detected


def test_drift_detected_shifted_distribution() -> None:
    reference = _make_df(500, seed=1)
    # Shift every column to a completely different range
    current = reference.copy()
    current["District"] = 25  # constant — no variance
    current["hour"] = 0
    current["month"] = 12
    current["Year"] = 2001
    current["Beat"] = 2500
    current["Latitude"] = 41.6
    current["Longitude"] = -87.9
    current["primary_type"] = "THEFT"
    current["location_description"] = "STREET"
    current["Community Area"] = 77
    detected, report = check_drift_from_dataframes(reference, current)
    assert detected
    assert report["share_of_drifted_columns"] > 0.2


def test_report_has_required_keys() -> None:
    ref = _make_df(200, seed=3)
    cur = _make_df(200, seed=4)
    _, report = check_drift_from_dataframes(ref, cur)
    assert "share_of_drifted_columns" in report
    assert "number_of_drifted_columns" in report
    assert "number_of_columns" in report
    assert "dataset_drift" in report
