"""Inferencia: carga el modelo desde DagsHub y predice probabilidad de arresto."""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.xgboost
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src.ml.features import FEATURE_COLS, build_single_record

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _load_model_uncached() -> tuple[xgb.Booster, LabelEncoder, LabelEncoder]:
    """Downloads the Production model + encoders from DagsHub MLflow registry.

    Returns:
        Tuple of (booster, le_primary, le_location).
    """
    import dagshub

    from src.config import load_settings

    cfg = load_settings()
    dagshub.auth.add_app_token(token=cfg.dagshub_token.get_secret_value())
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri.get_secret_value())

    logger.info("Loading model arrest-predictor@Production from DagsHub...")
    booster: xgb.Booster = mlflow.xgboost.load_model(
        "models:/arrest-predictor@Production"
    )

    client = mlflow.tracking.MlflowClient()
    version = client.get_model_version_by_alias("arrest-predictor", "Production")
    local_path = mlflow.artifacts.download_artifacts(
        run_id=version.run_id,
        artifact_path="encoders/label_encoders.pkl",
    )
    with Path(local_path).open("rb") as f:
        encoders: dict[str, LabelEncoder] = pickle.load(f)  # noqa: S301

    le_primary: LabelEncoder = encoders["primary_type"]
    le_location: LabelEncoder = encoders["location"]
    logger.info(
        "Model loaded — %d primary types, %d location types",
        len(le_primary.classes_),
        len(le_location.classes_),
    )
    return booster, le_primary, le_location


# Preserve the cached loader across importlib.reload() calls (e.g. during
# testing inside a patch context).  On the very first import the attribute does
# not exist yet, so we define it normally.  On subsequent reloads we reuse
# whatever is already bound in the module namespace — which may be a unittest
# mock when the module is reloaded inside a ``with patch(...)`` block.
_current_module = sys.modules.get(__name__)
if _current_module is None or not hasattr(_current_module, "load_production_model"):

    @st.cache_resource(ttl=3600)
    def load_production_model() -> tuple[xgb.Booster, LabelEncoder, LabelEncoder]:
        """Cached wrapper for Streamlit — reloads every hour."""
        return _load_model_uncached()

else:
    # Re-use the existing binding (mock or real cached function).
    load_production_model = _current_module.load_production_model  # type: ignore[assignment]


def predict_arrest_probability(record: dict[str, object]) -> float:
    """Returns arrest probability (0.0–1.0) for one crime record.

    Args:
        record: Dict with keys: primary_type, location_description, domestic,
            year, hour, district, community_area, latitude, beat, longitude,
            month, quarter, weekday.

    Returns:
        Float probability of arrest.
    """
    booster, le_primary, le_location = load_production_model()
    features = build_single_record(record, le_primary, le_location)
    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLS)
    prob = booster.predict(dmatrix)
    return float(prob[0])
