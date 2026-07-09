"""Modal retraining pipeline — drift-triggered XGBoost retraining con Optuna + MLflow."""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import optuna
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal image — paquetes necesarios dentro del contenedor GPU
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "xgboost>=3.2.0",
        "optuna>=4.9.0",
        "mlflow>=3.13.0",
        "dagshub",
        "scikit-learn>=1.8.0",
        "duckdb>=1.5.3",
        "pyarrow>=24.0.0",
        "evidently>=0.7.21",
        "pydantic-settings>=2.14.1",
        "pandas>=2.0.0",
    )
    .add_local_python_source("src")
)

_secrets = [modal.Secret.from_name("chicago-crimes-secrets")]

app = modal.App("chicago-crimes-retrain")


# ---------------------------------------------------------------------------
# Función de entrenamiento — GPU T4
# ---------------------------------------------------------------------------


def _configure_mlflow() -> None:
    """Set up DagsHub auth and MLflow tracking URI from settings."""
    import os

    import dagshub
    import mlflow

    from src.config import load_settings

    cfg = load_settings()
    token = cfg.dagshub_token.get_secret_value()
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.dagshub_username.get_secret_value()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    dagshub.auth.add_app_token(token=token)  # type: ignore[attr-defined]
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri.get_secret_value())
    mlflow.set_experiment("chicago-crimes-retraining")


def _optuna_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
) -> float:
    """Optuna objective: trains one XGBClassifier and logs to MLflow child run.

    Called from within an active MLflow parent run — creates a nested child run.
    """
    import mlflow
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from xgboost import XGBClassifier

    params: dict[str, object] = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # Parent run is already active (called from within retrain's with-block).
    # We only need to open the nested child run here.
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({k: v for k, v in params.items()
                           if k not in {"objective", "eval_metric", "tree_method",
                                        "device", "random_state"}})
        clf = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        score = float(roc_auc_score(y, proba))
        mlflow.log_metric("roc_auc_cv", score)

    return score


@app.function(
    gpu="T4",
    timeout=3600,
    image=_image,
    secrets=_secrets,
)
def retrain(force: bool = False) -> dict[str, object]:
    """Main retraining function — runs drift check then Optuna + MLflow."""
    import duckdb
    import mlflow
    import optuna as _optuna
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from src.config import load_settings
    from src.ml.drift import check_drift
    from src.ml.features import build_features

    _configure_mlflow()

    cfg = load_settings()
    parquet_url = f"s3://{cfg.r2_bucket_name}/crimes_full.parquet"

    # DuckDB connection with R2 credentials
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(
        "SET s3_endpoint=?",
        [f"{cfg.r2_account_id.get_secret_value()}.r2.cloudflarestorage.com"],
    )
    conn.execute("SET s3_access_key_id=?", [cfg.r2_access_key_id.get_secret_value()])
    conn.execute("SET s3_secret_access_key=?", [cfg.r2_secret_access_key.get_secret_value()])
    conn.execute("SET s3_region='auto';")
    conn.execute("SET s3_url_style='path';")

    # 1. Drift check
    if not force:
        drift_detected, drift_report = check_drift(parquet_url, conn)
        if not drift_detected:
            logger.info("No drift detected — skipping retraining.")
            return {"retrained": False, "drift_report": drift_report}
    else:
        drift_report = {"dataset_drift": False, "forced": True}

    # 2. Build features from full dataset
    logger.info("Building features from R2...")
    X, y, le_primary, le_location = build_features(parquet_url, conn)  # noqa: N806
    conn.close()

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    # 3. Optuna study with MLflow parent run
    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_params({"n_trials": 30, "force": force})
        mlflow.log_dict(drift_report, "drift_report.json")

        study = _optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _optuna_objective(trial, X_train, y_train),
            n_trials=30,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_cv_roc_auc = study.best_value
        logger.info("Best trial: ROC-AUC=%.4f params=%s", best_cv_roc_auc, best_params)

        # 4. Retrain best model on full train set
        final_clf = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        final_clf.fit(X_train, y_train)
        booster = final_clf.get_booster()

        # 5. Evaluate on held-out test set
        test_proba = final_clf.predict_proba(X_test)[:, 1]
        new_roc_auc = float(roc_auc_score(y_test, test_proba))
        mlflow.log_metrics({
            "roc_auc_cv_best": best_cv_roc_auc,
            "roc_auc_test": new_roc_auc,
        })

        # 6. Compare vs production model
        client = mlflow.tracking.MlflowClient()
        try:
            prod_version = client.get_model_version_by_alias("arrest-predictor", "Production")
            prod_run = client.get_run(prod_version.run_id or "")
            prod_roc_auc = float(prod_run.data.metrics.get("roc_auc_test", 0.0))
        except Exception as e:
            logger.warning("Could not fetch production model metrics: %s", e)
            prod_roc_auc = 0.0

        logger.info(
            "New ROC-AUC=%.4f vs Production ROC-AUC=%.4f",
            new_roc_auc,
            prod_roc_auc,
        )

        if new_roc_auc > prod_roc_auc:
            # 7. Save encoders + register model
            with tempfile.TemporaryDirectory() as tmpdir:
                enc_path = Path(tmpdir) / "label_encoders.pkl"
                enc_path.write_bytes(
                    pickle.dumps({"primary_type": le_primary, "location": le_location})
                )
                mlflow.log_artifact(str(enc_path), artifact_path="encoders")

            model_info = mlflow.xgboost.log_model(
                booster,
                artifact_path="model",
                registered_model_name="arrest-predictor",
            )
            version = model_info.registered_model_version
            client.set_registered_model_alias("arrest-predictor", "Production", str(version))
            logger.info("New model registered as Production (version %s)", version)
            promoted = True
        else:
            logger.info("New model not better than production — not promoting.")
            promoted = False

    return {
        "retrained": True,
        "promoted": promoted,
        "new_roc_auc": new_roc_auc,
        "prod_roc_auc": prod_roc_auc,
        "drift_report": drift_report,
    }


@app.function(
    schedule=modal.Cron("0 6 * * *"),
    image=_image,
    secrets=_secrets,
)
def scheduled_retrain() -> None:
    """Cron job — runs daily at 6am UTC."""
    retrain.remote(force=False)


@app.local_entrypoint()
def main(force: bool = False) -> None:
    """Local entrypoint: `modal run src/ml/train.py [--force]`."""
    result = retrain.remote(force=force)
    print(result)
