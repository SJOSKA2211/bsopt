"""
Consolidated ML Orchestrator for Black-Scholes Platform.

Provides a unified interface for:
- Synthetic and real data collection
- Model training (XGBoost, Neural Networks)
- Hyperparameter optimization (Optuna)
- MLflow tracking and model registration
"""

import asyncio
import orjson
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

import asyncio
import orjson
import structlog
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast, List

import click
import mlflow
import mlflow.data
import mlflow.pytorch
import mlflow.xgboost
from mlflow.tracking import MlflowClient 
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from anyio.to_thread import run_sync

from src.config import settings
from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import load_or_collect_data, run_hyperparameter_optimization

# Configure logging
logger = structlog.get_logger(__name__)

# Constants
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
PROJECT_NAME = settings.PROJECT_NAME
RESULTS_DIR = Path(settings.ML_RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MLOrchestrator:
    """
    Unified orchestrator for all ML workflows with async awareness.
    """

    def __init__(self):
        mlflow.set_tracking_uri(TRACKING_URI)
        os.makedirs("mlruns", exist_ok=True)

    def _get_experiment_name(self, model_type: str) -> str:
        return f"{PROJECT_NAME}/{model_type.upper()}_Training"

    async def run_training_pipeline(
        self,
        model_type: str = "xgboost",
        use_real_data: bool = False,
        n_samples: int = settings.ML_TRAINING_DEFAULT_SAMPLES,
        test_size: float = settings.ML_TRAINING_TEST_SIZE,
        random_state: int = settings.ML_TRAINING_RANDOM_STATE,
        xgb_params: Optional[Dict[str, Any]] = None,
        nn_params: Optional[Dict[str, Any]] = None,
        promotion_threshold_r2: float = settings.ML_TRAINING_PROMOTE_THRESHOLD_R2,
        promote_to_production: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a full training and evaluation pipeline asynchronously.
        """
        exp_name = self._get_experiment_name(model_type)
        mlflow.set_experiment(exp_name)
        
        client = MlflowClient()

        # 1. Data Preparation
        X, y, features, metadata = await load_or_collect_data(use_real_data, n_samples)
        
        # Offload scikit-learn split to thread
        X_train, X_test, y_train, y_test = await run_sync(
            train_test_split, X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = await run_sync(scaler.fit_transform, X_train)
        X_test_scaled = await run_sync(scaler.transform, X_test)

        # 2. Training
        with mlflow.start_run(run_name=f"{model_type}_training"):
            mlflow.set_tag("data_source", metadata.get("data_source", "unknown"))
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("run_date", datetime.now().strftime("%Y-%m-%d"))
            
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            
            registered_model_name = f"OptionPricer_{model_type.upper()}"

            if model_type == "xgboost":
                train_params = xgb_params or {
                    "max_depth": settings.ML_XGBOOST_MAX_DEPTH,
                    "learning_rate": settings.ML_XGBOOST_LEARNING_RATE,
                    "n_estimators": settings.ML_XGBOOST_N_ESTIMATORS,
                    "n_jobs": -1,
                    "objective": "reg:squarederror",
                }
                mlflow.log_params(train_params)
                model = xgb.XGBRegressor(**train_params)
                
                # Offload heavy XGBoost training to thread
                await run_sync(model.fit, X_train_scaled, y_train)
                mlflow.xgboost.log_model(model, "model", registered_model_name=registered_model_name)

            elif model_type == "nn":
                nn_train_params = nn_params or {
                    "lr": settings.ML_TRAINING_NN_LR,
                    "epochs": settings.ML_TRAINING_NN_EPOCHS,
                    "hidden_dims": settings.ML_TRAINING_NN_HIDDEN_DIMS,
                }
                mlflow.log_params(nn_train_params)
                
                hidden_dims = cast(list[int], nn_train_params.get("hidden_dims", settings.ML_TRAINING_NN_HIDDEN_DIMS))
                model = OptionPricingNN(input_dim=X.shape[1], hidden_dims=hidden_dims)
                optimizer = torch.optim.Adam(model.parameters(), lr=nn_train_params["lr"])
                criterion = torch.nn.MSELoss()

                model.train()
                epochs = nn_train_params["epochs"]
                
                # Move PyTorch training to thread to avoid event loop stalls
                def _train_loop():
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        output = model(torch.from_numpy(X_train_scaled).float())
                        loss = criterion(output, torch.from_numpy(y_train).float().view(-1, 1))
                        loss.backward()
                        optimizer.step()
                        mlflow.log_metric("loss", float(loss.item()), step=epoch)
                
                await run_sync(_train_loop)
                mlflow.pytorch.log_model(model, "model", registered_model_name=registered_model_name)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # 3. Evaluation
            if model_type == "xgboost":
                y_pred = await run_sync(model.predict, X_test_scaled)
            else:
                model.eval()
                with torch.no_grad():
                    # Evaluate in thread
                    def _eval():
                        return model(torch.from_numpy(X_test_scaled).float()).numpy().flatten()
                    y_pred = await run_sync(_eval)

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
                logger.info("model_promotion_triggered", name=registered_model_name, r2=r2, threshold=promotion_threshold_r2)
                
                if active_run:
                    model_versions = client.search_model_versions(f"run_id='{run_id}'")
                    if model_versions:
                        current_model_version = model_versions[0].version
                        client.transition_model_version_stage(
                            name=registered_model_name,
                            version=current_model_version,
                            stage="Production"
                        )
                        logger.info("model_promoted_to_production", name=registered_model_name, version=current_model_version)
                        mlflow.set_tag("model_stage", "Production")

                        # 5. Export to ONNX for WASM/High-speed inference
                        if model_type == "nn":
                            onnx_path = RESULTS_DIR / f"{registered_model_name}.onnx"
                            try:
                                dummy_input = torch.randn(1, X.shape[1])
                                torch.onnx.export(
                                    model,
                                    dummy_input,
                                    str(onnx_path),
                                    export_params=True,
                                    opset_version=12,
                                    do_constant_folding=True,
                                    input_names=['input'],
                                    output_names=['output'],
                                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                                )
                                logger.info("model_exported_to_onnx", path=str(onnx_path))
                                
                                # 6. Extreme Optimization: INT8 Quantization
                                try:
                                    from onnxruntime.quantization import quantize_dynamic, QuantType
                                    quantized_path = RESULTS_DIR / f"{registered_model_name}_int8.onnx"
                                    quantize_dynamic(
                                        str(onnx_path),
                                        str(quantized_path),
                                        weight_type=QuantType.QUInt8
                                    )
                                    logger.info("model_quantized_to_int8", path=str(quantized_path))
                                    mlflow.log_artifact(str(quantized_path), "quantized_model")
                                except ImportError:
                                    logger.warning("onnx_quantization_skipped", reason="onnxruntime-quantization not installed")
                                except Exception as e:
                                    logger.error("onnx_quantization_failed", error=str(e))

                                mlflow.log_artifact(str(onnx_path), "onnx_model")
                            except Exception as e:
                                logger.error("onnx_export_failed", error=str(e))
                    else:
                        logger.warning("model_version_not_found_for_promotion", run_id=run_id)
                else:
                    logger.warning("no_active_run_for_promotion")

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
    click.echo(f"Training completed: {orjson.dumps(result, option=orjson.OPT_INDENT_2).decode('utf-8')}")


@cli.command()
@click.option("--samples", type=int, default=5000)
@click.option("--trials", type=int, default=20)
def optimize(samples, trials):
    """Run hyperparameter optimization."""
    result = asyncio.run(run_hyperparameter_optimization(
        use_real_data=False, n_samples=samples, n_trials=trials
    ))
    click.echo(f"Optimization completed: {orjson.dumps(result, option=orjson.OPT_INDENT_2).decode('utf-8')}")


if __name__ == "__main__":
    cli()
