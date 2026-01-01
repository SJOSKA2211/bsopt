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
from typing import Any, Dict, Optional

import click
import mlflow
import mlflow.data
import mlflow.pytorch
import mlflow.xgboost
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{PROJECT_NAME}/{model_type.upper()}/{date_str}"

    async def run_training_pipeline(
        self,
        model_type: str = "xgboost",
        use_real_data: bool = False,
        n_samples: int = 10000,
        params: Optional[Dict[str, Any]] = None,
        promote: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a full training and evaluation pipeline.
        """
        exp_name = self._get_experiment_name(model_type)
        mlflow.set_experiment(exp_name)

        # 1. Data Preparation
        X, y, features, metadata = load_or_collect_data(use_real_data, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 2. Training
        with mlflow.start_run(run_name=f"{model_type}_training"):
            mlflow.log_params(params or {})
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("data_source", metadata.get("data_source", "unknown"))

            if model_type == "xgboost":
                train_params = params or {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "n_jobs": -1,
                }
                model = xgb.XGBRegressor(**train_params)
                model.fit(X_train_scaled, y_train)
                model._estimator_type = "regressor"
                mlflow.xgboost.log_model(model, "model", registered_model_name="OptionPricer_XGB")

            elif model_type == "nn":
                model = OptionPricingNN(input_dim=X.shape[1])
                # Simple training loop for brevity in orchestrator
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=params.get("lr", 0.001) if params else 0.001
                )
                criterion = torch.nn.MSELoss()

                model.train()
                epochs = params.get("epochs", 10) if params else 10
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    output = model(torch.FloatTensor(X_train_scaled))
                    loss = criterion(output, torch.FloatTensor(y_train).view(-1, 1))
                    loss.backward()
                    optimizer.step()
                    mlflow.log_metric("loss", float(loss.item()), step=epoch)

                mlflow.pytorch.log_model(model, "model", registered_model_name="OptionPricer_NN")

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # 3. Evaluation
            if model_type == "xgboost":
                y_pred = model.predict(X_test_scaled)
            else:
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.FloatTensor(X_test_scaled)).numpy().flatten()

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mse", mse)

            result = {
                "run_id": mlflow.active_run().info.run_id,
                "r2": r2,
                "mse": mse,
                "model_type": model_type,
            }

            # 4. Model Promotion
            if promote and r2 > 0.98:
                logger.info(f"Promoting model {model_type} to Production...")
                # Implementation depends on MLflow registry setup
                pass

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
    result = run_hyperparameter_optimization(
        use_real_data=False, n_samples=samples, n_trials=trials
    )
    click.echo(f"Optimization completed: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    cli()
