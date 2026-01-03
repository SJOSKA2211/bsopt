"""
Consolidated ML Orchestrator for Black-Scholes Platform.

Provides a unified interface for:
- Synthetic and real data collection
- Model training (XGBoost, Neural Networks)
- Hyperparameter optimization (Optuna)
- MLflow tracking and model registration
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast, List

import click
import mlflow
import mlflow.data
import mlflow.pytorch
import mlflow.xgboost
from mlflow.tracking import MlflowClient # Added for model promotion
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings

from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import load_or_collect_data, run_hyperparameter_optimization

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
PROJECT_NAME = "BlackScholesPlatform"
RESULTS_DIR = Path("results/ml")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MLOrchestrator:
    """
    Unified orchestrator for all ML workflows.
    """

    def __init__(self):
        mlflow.set_tracking_uri(TRACKING_URI)
        os.makedirs("mlruns", exist_ok=True)

    def _get_experiment_name(self, model_type: str) -> str:
        # Use a more stable experiment name, adding date as a tag to runs
        return f"{PROJECT_NAME}/{model_type.upper()}_Training"

    async def run_training_pipeline(
        self,
        model_type: str = "xgboost",
        use_real_data: bool = False,
        n_samples: int = 10000,
        test_size: float = settings.ML_TRAINING_TEST_SIZE, # Use setting
        random_state: int = settings.ML_TRAINING_RANDOM_STATE, # Use setting
        xgb_params: Optional[Dict[str, Any]] = None,
        nn_params: Optional[Dict[str, Any]] = None,
        promotion_threshold_r2: float = settings.ML_TRAINING_PROMOTE_THRESHOLD_R2, # Use setting
        promote_to_production: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a full training and evaluation pipeline.
        """
        exp_name = self._get_experiment_name(model_type)
        mlflow.set_experiment(exp_name)
        
        # MLflow client for model registry operations
        client = MlflowClient()

        # 1. Data Preparation
        X, y, features, metadata = await load_or_collect_data(use_real_data, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 2. Training
        with mlflow.start_run(run_name=f"{model_type}_training"):
            mlflow.set_tag("data_source", metadata.get("data_source", "unknown"))
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("run_date", datetime.now().strftime("%Y-%m-%d")) # Add run date as a tag
            
            # Log training parameters
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            
            registered_model_name = f"OptionPricer_{model_type.upper()}"

            if model_type == "xgboost":
                train_params = xgb_params or {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "n_jobs": -1,
                    "objective": "reg:squarederror", # Explicitly set objective
                }
                mlflow.log_params(train_params)
                model = xgb.XGBRegressor(**train_params)
                model.fit(X_train_scaled, y_train)
                # No need for model._estimator_type = "regressor", MLflow should infer from XGBRegressor
                model_info = mlflow.xgboost.log_model(model, "model", registered_model_name=registered_model_name)

            elif model_type == "nn":
                nn_train_params = nn_params or {
                    "lr": settings.ML_TRAINING_NN_LR, # Use setting
                    "epochs": settings.ML_TRAINING_NN_EPOCHS, # Use setting
                    "hidden_dims": [128, 64, 32],
                }
                mlflow.log_params(nn_train_params)
                
                hidden_dims = cast(list[int], nn_train_params.get("hidden_dims", [128, 64, 32]))
                model = OptionPricingNN(input_dim=X.shape[1], hidden_dims=hidden_dims)
                optimizer = torch.optim.Adam(model.parameters(), lr=nn_train_params["lr"])
                criterion = torch.nn.MSELoss()

                model.train()
                epochs = nn_train_params["epochs"]
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    output = model(torch.from_numpy(X_train_scaled).float()) # Use from_numpy
                    loss = criterion(output, torch.from_numpy(y_train).float().view(-1, 1)) # Use from_numpy
                    loss.backward()
                    optimizer.step()
                    mlflow.log_metric("loss", float(loss.item()), step=epoch)

                model_info = mlflow.pytorch.log_model(model, "model", registered_model_name=registered_model_name)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # 3. Evaluation
            if model_type == "xgboost":
                y_pred = model.predict(X_test_scaled)
            else:
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.from_numpy(X_test_scaled).float()).numpy().flatten() # Use from_numpy

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mse", mse)

            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else "unknown"

            result = {
                "run_id": run_id,
                "r2": r2,
                "mse": mse,
                "model_type": model_type,
            }

            # 4. Model Promotion
            if promote_to_production and r2 > promotion_threshold_r2:
                logger.info(f"Model {registered_model_name} (Run ID: {result['run_id']}) "
                            f"achieved R2 of {r2:.4f}, which is above threshold {promotion_threshold_r2:.4f}.")
                # Get the latest version of the registered model
                latest_versions = client.search_model_versions(f"name='{registered_model_name}'")
                latest_version = max([int(mv.version) for mv in latest_versions]) if latest_versions else 1
                
                # Transition new model version to Production stage
                # Need to find the ModelVersion object corresponding to the current run
                if active_run:
                    model_versions = client.search_model_versions(f"run_id='{run_id}'")
                    if model_versions:
                        current_model_version = model_versions[0].version
                        client.transition_model_version_stage(
                            name=registered_model_name,
                            version=current_model_version,
                            stage="Production"
                        )
                        logger.info(f"Model '{registered_model_name}' version {current_model_version} "
                                    f"transitioned to Production stage.")
                        mlflow.set_tag("model_stage", "Production")
                    else:
                        logger.warning(f"Could not find model version for run {run_id} to promote.")
                else:
                    logger.warning("No active MLflow run found for promotion.")

            return result


@click.group()
def cli():
    """ML Orchestration CLI."""
    pass


@cli.command()
@click.option("--model", type=click.Choice(["xgboost", "nn"]), default="xgboost")
@click.option("--samples", type=int, default=5000)
@click.option("--real-data", is_flag=True, help="Use real market data if available")
def train(model, samples, real_data):
    """Run model training."""
    orchestrator = MLOrchestrator()
    result = asyncio.run(
        orchestrator.run_training_pipeline(
            model_type=model, use_real_data=real_data, n_samples=samples
        )
    )
    click.echo(f"Training completed: {json.dumps(result, indent=2)}")


@cli.command()
@click.option("--samples", type=int, default=5000)
@click.option("--trials", type=int, default=20)
def optimize(samples, trials):
    """Run hyperparameter optimization."""
    result = asyncio.run(run_hyperparameter_optimization(
        use_real_data=False, n_samples=samples, n_trials=trials
    ))
    click.echo(f"Optimization completed: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    cli()
