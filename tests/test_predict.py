from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def sample_record() -> dict:
    return {
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


def _make_mock_encoders():
    le_primary = MagicMock()
    le_primary.transform.return_value = np.array([5])
    le_primary.classes_ = np.array(["ARSON", "BATTERY", "BURGLARY", "HOMICIDE",
                                     "ROBBERY", "THEFT"])

    le_location = MagicMock()
    le_location.transform.return_value = np.array([10])
    le_location.classes_ = np.array(["ALLEY", "APARTMENT", "RESIDENCE", "SIDEWALK",
                                      "STREET"])
    return le_primary, le_location


def test_predict_returns_float_in_range(sample_record: dict) -> None:
    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([0.72])
    le_primary, le_location = _make_mock_encoders()

    with patch("src.ml.predict.load_production_model",
               return_value=(mock_booster, le_primary, le_location)):
        from src.ml.predict import predict_arrest_probability
        result = predict_arrest_probability(sample_record)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert abs(result - 0.72) < 1e-6


def test_predict_calls_booster_predict(sample_record: dict) -> None:
    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([0.5])
    le_primary, le_location = _make_mock_encoders()

    with patch("src.ml.predict.load_production_model",
               return_value=(mock_booster, le_primary, le_location)):
        import importlib

        import src.ml.predict as predict_mod
        importlib.reload(predict_mod)
        predict_mod.predict_arrest_probability(sample_record)

    mock_booster.predict.assert_called_once()


def test_predict_night_record_runs(sample_record: dict) -> None:
    sample_record["hour"] = 23  # night hour
    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([0.85])
    le_primary, le_location = _make_mock_encoders()

    with patch("src.ml.predict.load_production_model",
               return_value=(mock_booster, le_primary, le_location)):
        from src.ml.predict import predict_arrest_probability
        result = predict_arrest_probability(sample_record)

    assert 0.0 <= result <= 1.0
