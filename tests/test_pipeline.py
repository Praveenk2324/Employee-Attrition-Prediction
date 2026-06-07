"""
Run with: pytest tests/ -v
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline


SAMPLE_INPUT = {
    "Age": 35,
    "Department": "Sales",
    "JobRole": "Sales Executive",
    "JobSatisfaction": 2,
    "EnvironmentSatisfaction": 2,
    "WorkLifeBalance": 3,
    "JobInvolvement": 3,
    "PerformanceRating": 3,
    "OverTime": "Yes",
    "Gender": "Male",
    "MaritalStatus": "Single",
    "EducationField": "Life Sciences",
    "BusinessTravel": "Travel_Frequently",
    "Education": 3,
    "JobLevel": 2,
    "MonthlyIncome": 4500,
    "DailyRate": 800,
    "HourlyRate": 65,
    "MonthlyRate": 14000,
    "DistanceFromHome": 10,
    "NumCompaniesWorked": 3,
    "PercentSalaryHike": 13,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 2,
    "YearsAtCompany": 3,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
    "RelationshipSatisfaction": 3,
}


class TestPipelineBuild:
    def test_pipeline_has_two_steps(self):
        from pipeline import build_pipeline
        cfg = _minimal_config()
        pipe = build_pipeline(cfg)
        assert isinstance(pipe, Pipeline)
        assert list(pipe.named_steps.keys()) == ["preprocessor", "classifier"]

    def test_classifier_params_match_config(self):
        from pipeline import build_pipeline
        cfg = _minimal_config()
        pipe = build_pipeline(cfg)
        clf = pipe.named_steps["classifier"]
        assert clf.n_estimators == cfg["model"]["n_estimators"]
        assert clf.max_depth == cfg["model"]["max_depth"]


class TestPredictOutput:
    def test_predict_returns_required_keys(self):
        from predict import predict
        pipeline, explainer, config = _mock_artifacts()
        result = predict(SAMPLE_INPUT, pipeline, explainer, config)
        assert "attrition_probability" in result
        assert "prediction" in result
        assert "shap_explanation" in result
        assert "threshold_used" in result

    def test_probability_is_between_0_and_1(self):
        from predict import predict
        pipeline, explainer, config = _mock_artifacts()
        result = predict(SAMPLE_INPUT, pipeline, explainer, config)
        assert 0.0 <= result["attrition_probability"] <= 1.0

    def test_prediction_label_is_yes_or_no(self):
        from predict import predict
        pipeline, explainer, config = _mock_artifacts()
        result = predict(SAMPLE_INPUT, pipeline, explainer, config)
        assert result["prediction"] in ["Yes", "No"]

    def test_shap_explanation_has_correct_length(self):
        from predict import predict
        pipeline, explainer, config = _mock_artifacts()
        result = predict(SAMPLE_INPUT, pipeline, explainer, config)
        assert len(result["shap_explanation"]) == config["inference"]["shap_top_n"]

    def test_threshold_applied_correctly(self):
        from predict import predict
        pipeline, explainer, config = _mock_artifacts(prob=0.8)
        result = predict(SAMPLE_INPUT, pipeline, explainer, config)
        assert result["prediction"] == "Yes"

        pipeline2, explainer2, config2 = _mock_artifacts(prob=0.3)
        result2 = predict(SAMPLE_INPUT, pipeline2, explainer2, config2)
        assert result2["prediction"] == "No"


class TestSchema:
    def test_valid_input_passes(self):
        from api.schema import EmployeeInput
        emp = EmployeeInput(**SAMPLE_INPUT)
        assert emp.Age == 35

    def test_invalid_overtime_fails(self):
        from api.schema import EmployeeInput
        from pydantic import ValidationError
        bad = {**SAMPLE_INPUT, "OverTime": "Maybe"}
        with pytest.raises(ValidationError):
            EmployeeInput(**bad)

    def test_age_out_of_range_fails(self):
        from api.schema import EmployeeInput
        from pydantic import ValidationError
        bad = {**SAMPLE_INPUT, "Age": 15}
        with pytest.raises(ValidationError):
            EmployeeInput(**bad)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _minimal_config() -> dict:
    return {
        "model": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "early_stopping_rounds": 10
        },
        "features": {
            "onehot":  ["Department", "JobRole", "MaritalStatus", "EducationField", "BusinessTravel"],
            "ordinal": ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
                        "JobInvolvement", "PerformanceRating"],
            "binary":  ["OverTime", "Gender"],
            "numeric": ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
                        "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "StockOptionLevel",
                        "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
                        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
                        "Education", "JobLevel", "RelationshipSatisfaction"],
        },
    }


def _mock_artifacts(prob: float = 0.75):
    """Returns (mock_pipeline, mock_explainer, config) for unit tests."""
    n_features = 50
    config = {
        "inference": {"threshold": 0.60, "shap_top_n": 3},
    }

    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[1 - prob, prob]])

    mock_pre = MagicMock()
    mock_pre.transform.return_value = np.zeros((1, n_features))
    mock_pre.get_feature_names_out.return_value = [f"feat_{i}" for i in range(n_features)]
    mock_pipeline.named_steps = {"preprocessor": mock_pre}

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [np.random.randn(n_features)]

    return mock_pipeline, mock_explainer, config