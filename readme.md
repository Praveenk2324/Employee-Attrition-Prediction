# Attrition MLOps Pipeline

XGBoost employee attrition prediction — end-to-end MLOps project with:
- **sklearn Pipeline** (preprocessor + classifier as one object)
- **DVC** for data and pipeline versioning
- **MLflow** for experiment tracking and model registry
- **FastAPI** for serving predictions with SHAP explanations

---

## Project structure

```
attrition-mlops/
├── config/
│   └── config.yaml          # all hyperparams and paths (no hardcoding in src/)
├── data/
│   ├── raw/                 # attrition.csv tracked by DVC
│   └── processed/           # train.csv / test.csv (DVC outputs)
├── src/
│   ├── data_ingestion.py    # load CSV, drop constants, stratified split
│   ├── pipeline.py          # build ColumnTransformer + XGBClassifier Pipeline
│   ├── train.py             # orchestrator — DVC stage entry point + MLflow run
│   ├── evaluate.py          # metrics, threshold tuning, reports/metrics.json
│   └── predict.py           # load pipeline.pkl, run inference + SHAP
├── api/
│   ├── main.py              # FastAPI app — loads pipeline once at startup
│   └── schema.py            # Pydantic input model for all 32 fields
├── tests/
│   └── test_pipeline.py     # unit tests for pipeline, predict, schema
├── artifacts/               # pipeline.pkl (DVC output, gitignored)
├── reports/
│   └── metrics.json         # DVC metric file (committed)
├── dvc.yaml                 # DVC pipeline stage definitions    
|
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Get WA_Fn-UseC_-HR-Employee-Attrition.csv from Kaggle
# Place at: data/raw/attrition.csv

# 4. Initialise git and DVC
git init
dvc init

# 5. Track raw data with DVC (commits a pointer, not the CSV)
dvc add data/raw/attrition.csv
git add data/raw/attrition.csv.dvc .gitignore
git commit -m "track raw data with dvc"

# 6. (Optional) Add a DVC remote for team sharing
dvc remote add -d myremote s3://your-bucket/attrition-dvc
# or local:
dvc remote add -d myremote /tmp/dvcstore
```

---

## Running the pipeline

```bash
# Run the full DVC pipeline (ingest → train → evaluate)
dvc repro

# Check what changed since the last run
dvc params diff        # hyperparameter changes
dvc metrics diff       # roc_auc, f1 before vs after

# Push data and artifacts to DVC remote
dvc push
```

---

## Experiment tracking with MLflow

```bash
# Launch the MLflow UI (after at least one dvc repro)
mlflow ui

# Open: http://localhost:5000
# You'll see all runs with params, metrics, and the registered pipeline
```

To tune and compare runs, change values in `config/config.yaml` (e.g. bump `max_depth` to 6), then run `dvc repro` again. MLflow records each run separately.

To promote a run to production in MLflow:
1. Open the MLflow UI
2. Go to Models → attrition-pipeline
3. Transition the version you want to "Production"

---

## Serving predictions with FastAPI

```bash
# Start the API server
uvicorn api.main:app --reload

# Interactive docs
open http://localhost:8000/docs
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "RelationshipSatisfaction": 3
  }'
```

Example response:

```json
{
  "attrition_probability": 0.7842,
  "prediction": "Yes",
  "threshold_used": 0.6,
  "shap_explanation": [
    {"feature": "bin__OverTime_Yes",  "shap_value": 0.42, "direction": "increases risk"},
    {"feature": "num__MonthlyIncome", "shap_value": 0.28, "direction": "increases risk"},
    {"feature": "ord__JobSatisfaction","shap_value": 0.21, "direction": "increases risk"}
  ]
}
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Key design decisions

**Single Pipeline object** — `ColumnTransformer` and `XGBClassifier` are wrapped in one `sklearn.Pipeline`. The API receives raw field values; no manual `.transform()` call is needed at inference. SHAP accesses the classifier via `pipeline.named_steps["classifier"]`.

**DVC owns data and artifact versioning** — `data/raw/attrition.csv` is tracked by DVC (pointer in git, file in remote storage). `artifacts/pipeline.pkl` is a DVC output. If config changes, `dvc repro` re-runs only the stale stages.

**MLflow owns experiment tracking** — every `dvc repro` run opens an MLflow run, logs all params and metrics, and registers the pipeline to the model registry under `attrition-pipeline`.

**No hardcoded values anywhere** — all hyperparameters, paths, thresholds, and feature lists live in `config/config.yaml`. Change one file, re-run, compare in MLflow.