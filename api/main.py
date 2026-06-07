"""
FastAPI application.
Start: uvicorn api.main:app --reload
Docs:  http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append("src")

from predict import load_config, load_artifacts, predict as run_predict
from api.schema import EmployeeInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Module-level state (loaded once at startup) ─────────────────────────────
_pipeline  = None
_explainer = None
_config    = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline and SHAP explainer once when server starts."""
    global _pipeline, _explainer, _config
    logger.info("Loading pipeline and SHAP explainer...")
    _config    = load_config()
    _pipeline, _explainer = load_artifacts(_config)
    logger.info("Ready to serve predictions.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Employee Attrition Prediction API",
    description=(
        "XGBoost + sklearn Pipeline model for predicting employee attrition. "
        "Returns probability, decision, and SHAP-based explanation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness check — confirms pipeline is loaded."""
    return {
        "status":   "ok",
        "pipeline": _pipeline is not None,
    }


@app.post("/predict", response_model=dict[str, Any])
def predict(employee: EmployeeInput) -> dict:
    """
    Predict attrition probability for one employee.

    - Accepts all 32 employee fields (see schema for allowed values).
    - Returns probability, binary prediction, and top SHAP drivers.
    - Preprocessing is handled internally by the pipeline — no encoding needed by the caller.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = run_predict(
            input_dict=employee.model_dump(),
            pipeline=_pipeline,
            explainer=_explainer,
            config=_config,
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Attrition Prediction API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }