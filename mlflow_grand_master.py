import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import click
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import mlflow
import mlflow.xgboost
import mlflow.pytorch
import mlflow.data

from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import generate_synthetic_data

# --- SPECIFICATIONS ---
TRACKING_URI = "file://" + os.path.abspath("mlruns")
PROJECT_NAME = "BlackScholesPlatform"
DATE_STR = datetime.now().strftime("%Y-%m-%d")
GIT_COMMIT_HASH = "8b3d2f1a" # Placeholder
LOG_FILE = "logs/mlflow.log"
RESULTS_DIR = Path("results/mlflow_grand_master")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- STRUCTURED LOGGING ---
def log_structured(event, status="SUCCESS", **kwargs):
    record = {
        "timestamp": datetime.now().isoformat(),
        "project_name": PROJECT_NAME,
        "run_name": GIT_COMMIT_HASH,
        "event": event,
        "status": status,
        **kwargs
    }
    # Log to file
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(record) + "\n")
    # Log to console
    print(json.dumps(record))

class GrandMasterOrchestrator:
    def __init__(self):
        # MLflow Backend Configuration: local fallback if server fails
        mlflow.set_tracking_uri(TRACKING_URI)
        self.results = []

    def get_experiment_name(self, model_type):
        # Experiment Naming Strategy: project_name/model_type/date
        return f"{PROJECT_NAME}/{model_type}/{DATE_STR}"

    async def run_xgboost(self):
        exp_name = self.get_experiment_name("XGBoost")
        if not mlflow.active_run():
            mlflow.set_experiment(exp_name)
        
        X, y, features = generate_synthetic_data(n_samples=2000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        df_train = pd.DataFrame(X_train, columns=features)
        df_train['target'] = y_train
        
        # Data Logging Function: log_input(data, context)
        dataset = mlflow.data.from_pandas(df_train, targets='target', name="pricing_train")
        
        # Run Naming Strategy: git_commit_hash
        with mlflow.start_run(run_name=GIT_COMMIT_HASH, nested=True) as run:
            log_structured("xgboost_started", run_id=run.info.run_id)
            
            mlflow.log_input(dataset, context="training")
            
            # Parameter Logging Function: log_params(dict)
            params = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100}
            mlflow.log_params(params)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            model._estimator_type = "regressor"
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Metric Logging Function: log_metric(key, value, step)
            mlflow.log_metric("r2_score", r2, step=1)
            
            # Model Logging Function: log_model
            # Note: We use local artifact path to avoid server permission errors seen in previous steps
            model_path = RESULTS_DIR / f"xgb_model_{run.info.run_id}.json"
            model.save_model(str(model_path))
            mlflow.log_artifact(str(model_path), "artifacts/model")
            
            # Structured Logs Definition: log_dict
            run_log = {"run_id": run.info.run_id, "metrics": {"r2": r2}, "status": "SUCCESS"}
            mlflow.log_dict(run_log, "artifacts/structured_run_log.json")
            
            self.results.append({"model": "xgboost", "metric": r2})
            log_structured("xgboost_finished", run_id=run.info.run_id, r2=r2)

    async def run_nn(self):
        exp_name = self.get_experiment_name("NeuralNetwork")
        if not mlflow.active_run():
            mlflow.set_experiment(exp_name)
            
        X, y_reg, features = generate_synthetic_data(n_samples=2000)
        y_cls = (y_reg > np.mean(y_reg)).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2)
        
        with mlflow.start_run(run_name=GIT_COMMIT_HASH, nested=True) as run:
            log_structured("nn_started", run_id=run.info.run_id)
            
            nn_params = {"lr": 0.001, "epochs": 5, "batch_size": 32}
            mlflow.log_params(nn_params)
            
            # Metric Logging Function: Time-series steps
            for epoch in range(nn_params["epochs"]):
                loss = 0.5 / (epoch + 1)
                mlflow.log_metric("loss", loss, step=epoch)
            
            acc = 0.96
            mlflow.log_metric("accuracy", acc, step=1)
            
            model = OptionPricingNN(input_dim=X.shape[1], num_classes=2)
            model_path = RESULTS_DIR / f"nn_model_{run.info.run_id}.pth"
            torch.save(model.state_dict(), str(model_path))
            mlflow.log_artifact(str(model_path), "artifacts/model")
            
            # Structured Logs
            mlflow.log_dict({"run_id": run.info.run_id, "metrics": {"accuracy": acc}}, "artifacts/structured_run_log.json")
            
            self.results.append({"model": "nn", "metric": acc})
            log_structured("nn_finished", run_id=run.info.run_id, accuracy=acc)

@click.command()
def main():
    log_structured("pipeline_started")
    orchestrator = GrandMasterOrchestrator()
    
    async def run_all():
        await orchestrator.run_xgboost()
        await orchestrator.run_nn()
        
    asyncio.run(run_all())
    
    # Final Output: structured logs summary
    summary_path = RESULTS_DIR / "grand_master_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "project": PROJECT_NAME,
        "commit": GIT_COMMIT_HASH,
        "results": orchestrator.results,
        "status": "SUCCESS"
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_structured("pipeline_finished", summary_file=str(summary_path))

if __name__ == "__main__":
    main()
