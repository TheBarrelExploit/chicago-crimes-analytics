"""One-time script: registers models/xgboost_model.json to DagsHub MLflow registry.

Usage:
    uv run python -m src.ml.upload_initial
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path

import duckdb
import mlflow
import mlflow.xgboost
import xgboost as xgb

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/xgboost_model.json")


def upload_initial_model() -> None:
    """Fits encoders from R2, then registers the local model to DagsHub."""
    import dagshub

    from src.config import load_settings
    from src.ml.features import build_features

    cfg = load_settings()

    # Auth — add_app_token populates DagsHub's token cache; the env vars feed
    # MLflow's REST client with HTTP Basic Auth for the same credentials.
    import os

    dagshub.auth.add_app_token(token=cfg.dagshub_token.get_secret_value())  # type: ignore[attr-defined]
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.dagshub_username.get_secret_value()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.dagshub_token.get_secret_value()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri.get_secret_value())
    mlflow.set_experiment("chicago-crimes-retraining")

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {_MODEL_PATH}")

    logger.info("Loading local model from %s", _MODEL_PATH)
    booster = xgb.Booster()
    booster.load_model(str(_MODEL_PATH))

    # Connect to R2 to fit encoders on full dataset
    logger.info("Fitting LabelEncoders from R2 parquet...")
    parquet_url = f"s3://{cfg.r2_bucket_name}/crimes_full.parquet"
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(
        "SET s3_endpoint=?",
        [f"{cfg.r2_account_id.get_secret_value()}.r2.cloudflarestorage.com"],
    )
    conn.execute("SET s3_access_key_id=?", [cfg.r2_access_key_id.get_secret_value()])
    conn.execute(
        "SET s3_secret_access_key=?", [cfg.r2_secret_access_key.get_secret_value()]
    )
    conn.execute("SET s3_region='auto';")
    conn.execute("SET s3_url_style='path';")

    _, _, le_primary, le_location = build_features(parquet_url, conn)
    conn.close()
    logger.info(
        "Encoders fitted: %d primary types, %d location types",
        len(le_primary.classes_),
        len(le_location.classes_),
    )

    # Register to MLflow
    with mlflow.start_run(run_name="initial_upload") as run:
        mlflow.log_metrics(
            {
                "roc_auc_test": 0.8955,
                "accuracy": 0.8477,
                "f1": 0.7071,
            }
        )
        mlflow.log_params({"source": "initial_upload", "model_file": str(_MODEL_PATH)})

        with tempfile.TemporaryDirectory() as tmp_dir:
            enc_path = Path(tmp_dir) / "label_encoders.pkl"
            with enc_path.open("wb") as f:
                pickle.dump({"primary_type": le_primary, "location": le_location}, f)
            mlflow.log_artifact(str(enc_path), artifact_path="encoders")

            model_info = mlflow.xgboost.log_model(
                booster,
                artifact_path="model",
                registered_model_name="arrest-predictor",
            )
            version = model_info.registered_model_version

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias("arrest-predictor", "Production", str(version))
    logger.info(
        "Registered arrest-predictor v%s as Production in DagsHub (run_id=%s)",
        version,
        run.info.run_id,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    upload_initial_model()
