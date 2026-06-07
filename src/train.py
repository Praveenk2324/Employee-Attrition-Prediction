import joblib
import pandas as pd
import numpy as np
import mlflow
import logging
import yaml
from pathlib import Path
from sklearn.model_selection import cross_val_score

from ingest import ingest, load_config
from pipeline import build_pipeline
from evaluate import evaluate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def train(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)

    logger.info("--------Stage 1: Data Ingestion--------")
    ingest(cfg)

    target = cfg['data']['target_column']
    train_df = pd.read_csv(cfg['paths']['train_data'])
    test_df = pd.read_csv(cfg['paths']['test_data'])

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    with mlflow.start_run() as run:
        logger.info(f"Mlflow run: {run.info.run_id}")

        logger.info("------Stage 2: Training pipeline-------")

        pipeline = build_pipeline(cfg)

        pre = pipeline.named_steps['preprocessor']
        clf = pipeline.named_steps['classifier']

        X_train_enc = pre.fit_transform(X_train, y_train)
        X_test_enc = pre.transform(X_test)

        clf.fit(
            X_train_enc, y_train,
            eval_set=[(X_test_enc, y_test)],
            verbose=False,
            
        )

        logger.info(f"Best iteration: {clf.best_iteration}")

        logger.info("=== Stage 3: Cross-validation ===")

        cv_pipeline = build_pipeline(cfg)
         
        # 2. Disable early stopping dynamically (because CV doesn't pass an eval_set)
        cv_pipeline.set_params(classifier__early_stopping_rounds=None)

        # 3. Run cross validation using the modified pipeline
        cv_scores = cross_val_score(
            cv_pipeline, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )

        logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        logger.info("=== Stage 4: Evaluation ===")
        metrics = evaluate(
            pipeline,
            X_test, y_test,
            threshold=cfg['inference']['threshold'],
            metrics_path=cfg['paths']['metrics']
        )

        mlflow.log_params(cfg["model"])
        mlflow.log_param("threshold", cfg["inference"]["threshold"])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("best_iteration", clf.best_iteration)
 
        # Metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
 
        # Artifacts
        mlflow.log_artifact(cfg["paths"]["metrics"])
        mlflow.log_artifact("config/config.yaml")
 
        # Log the full pipeline as a sklearn model (single artifact)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="pipeline",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )
        logger.info("Pipeline logged to MLflow model registry")

        Path(cfg["paths"]["pipeline"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, cfg["paths"]["pipeline"])
        logger.info(f"Pipeline saved to {cfg['paths']['pipeline']}")

        logger.info("Generating feature importance...")
        
        # 1. Extract the preprocessor and classifier from the trained pipeline
        pre = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        
        # 2. Get the actual feature names (after one-hot encoding!)
        feature_names = pre.get_feature_names_out()
        
        # 3. Get the importance scores from XGBoost
        importances = clf.feature_importances_
        
        # 4. Create a DataFrame and sort it
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        # 5. Ensure the directory exists, then save the CSV
        Path("reports").mkdir(parents=True, exist_ok=True)
        importance_df.to_csv("reports/feature_importance.csv", index=False)
        
        logger.info("Feature importance saved to reports/feature_importance.csv")
 
        logger.info("=== Training complete ===")
        logger.info(f"  ROC-AUC:  {metrics['roc_auc']}")
        logger.info(f"  F1:       {metrics['f1_score']}")
        logger.info(f"  Recall:   {metrics['recall']}")
        logger.info(f"  Run ID:   {run.info.run_id}")

    return pipeline, metrics


if __name__=="__main__":
    train()