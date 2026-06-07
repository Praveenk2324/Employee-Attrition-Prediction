"""
Inference module. Loaded once at FastAPI startup.
Uses the single pipeline.pkl — no manual .transform() needed.
SHAP accesses the classifier via pipeline.named_steps.
"""

import joblib
import shap
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_artifacts(config: dict):
    """Load pipeline once at startup. Returns (pipeline, explainer, config)."""
    pipeline_path = config["paths"]["pipeline"]

    if not Path(pipeline_path).exists():
        raise FileNotFoundError(
            f"Pipeline not found at {pipeline_path}. Run `dvc repro` first."
        )

    pipeline = joblib.load(pipeline_path)
    logger.info(f"Pipeline loaded from {pipeline_path}")

    # SHAP TreeExplainer needs the raw XGBoost model (already fitted)
    classifier = pipeline.named_steps["classifier"]
    explainer  = shap.TreeExplainer(classifier)
    logger.info("SHAP TreeExplainer initialised")

    return pipeline, explainer


def predict(
    input_dict: dict,
    pipeline,
    explainer,
    config: dict,
) -> dict:
    """
    Full inference:
      1. raw dict → single-row DataFrame
      2. pipeline.predict_proba()  — preprocessor + model in one call
      3. SHAP via preprocessor.transform() → named_steps explainer
    """
    threshold = config["inference"]["threshold"]
    top_n     = config["inference"]["shap_top_n"]

    df = pd.DataFrame([input_dict])

    # ── Prediction ─────────────────────────────────────────────────────────
    prob = float(pipeline.predict_proba(df)[0][1])

    # ── SHAP explanation ────────────────────────────────────────────────────
    preprocessor = pipeline.named_steps["preprocessor"]
    X_enc        = preprocessor.transform(df)           # only for SHAP
    shap_vals    = explainer.shap_values(X_enc)[0]
    feat_names   = preprocessor.get_feature_names_out()

    # Top N by absolute SHAP value
    ranked = sorted(
        zip(feat_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    shap_explanation = [
        {
            "feature":    feat,
            "shap_value": round(float(val), 4),
            "direction":  "increases risk" if val > 0 else "reduces risk",
        }
        for feat, val in ranked
    ]

    return {
        "attrition_probability": round(prob, 4),
        "prediction":            "Yes" if prob >= threshold else "No",
        "threshold_used":        threshold,
        "shap_explanation":      shap_explanation,
    }