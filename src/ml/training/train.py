import asyncio
import os
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import structlog
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

from src.config import settings
from src.ml.evaluation.metrics import calculate_regression_metrics
from src.ml.training.data_gen import generate_synthetic_data_numba

logger = structlog.get_logger(__name__)


import torch
import torch.distributed as dist


def init_collective_backend():
    """Initialize NCCL backend for multi-GPU training if available."""
    if not torch.cuda.is_available():
        return
        
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method="env://"
            )
            logger.info("nccl_backend_initialized", world_size=dist.get_world_size())
    except Exception as e:
        logger.warning("nccl_init_failed", error=str(e))
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", init_method="env://")

def generate_synthetic_data(n_samples: int = settings.ML_TRAINING_DEFAULT_SAMPLES) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic training data using Numba-optimized Black-Scholes engine."""
    logger.info("generating_synthetic_data", n_samples=n_samples)
    return generate_synthetic_data_numba(n_samples=n_samples, random_state=settings.ML_TRAINING_RANDOM_STATE)


def objective(trial: optuna.Trial, x_vals: np.ndarray, y_vals: np.ndarray, n_folds: int = 3, is_timeseries: bool = True) -> float:
    """Optuna objective for XGBoost optimization."""
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": settings.ML_TRAINING_RANDOM_STATE,
        "n_jobs": -1,
    }

    if is_timeseries:
        cv = TimeSeriesSplit(n_splits=n_folds)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=settings.ML_TRAINING_RANDOM_STATE)
        
    scores = []
    for train_idx, val_idx in cv.split(x_vals):
        x_t, x_v = x_vals[train_idx], x_vals[val_idx]
        y_t, y_v = y_vals[train_idx], y_vals[val_idx]
        
        # Use sample weights in training to match weighted metrics logic
        weights = np.maximum(y_t, 1.0)
        
        model = xgb.XGBRegressor(**param)
        model.fit(x_t, y_t, sample_weight=weights, eval_set=[(x_v, y_v)], verbose=False)
        
        preds = model.predict(x_v)
        metrics = calculate_regression_metrics(y_v, preds)
        scores.append(metrics["r2"]) 
    
    res = float(np.mean(scores))
    logger.debug("trial_complete", r2=res, params=param)
    return res


from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


async def run_hyperparameter_optimization(
    use_real_data: bool = True, n_samples: int = settings.ML_TRAINING_DEFAULT_SAMPLES, n_trials: int = settings.ML_TRAINING_OPTUNA_TRIALS
) -> dict[str, Any]:
    """Distributed HPO using Ray Tune and Optuna."""
    x_vals, y_vals, _, _ = await load_or_collect_data(use_real_data=use_real_data, n_samples=n_samples)
    
    config = {
        "n_estimators": tune.randint(50, 1000),
        "max_depth": tune.randint(3, 15),
        "learning_rate": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
    }

    scheduler = ASHAScheduler(metric="r2", mode="max", max_t=100, grace_period=10)
    search_alg = OptunaSearch(metric="r2", mode="max")

    def trainable(config):
        """Inner trainable for Ray workers."""
        model = xgb.XGBRegressor(**config, n_jobs=1)
        # Use a simple split for the parallel trials
        x_t, x_v, y_t, y_v = train_test_split(x_vals, y_vals, test_size=0.2, shuffle=False)
        model.fit(x_t, y_t)
        preds = model.predict(x_v)
        r2 = r2_score(y_v, preds)
        tune.report(r2=r2)

    logger.info("starting_distributed_hpo", n_trials=n_trials)
    
    analysis = tune.run(
        trainable,
        config=config,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        verbose=1
    )
    
    best_config = analysis.get_best_config(metric="r2", mode="max")
    logger.info("distributed_hpo_complete", best_r2=analysis.best_result["r2"])
    return {"best_params": best_config, "best_r2": analysis.best_result["r2"]}


async def collect_real_data(
    symbols: list[str] | None = None, min_samples: int = 10000
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Collect options data from market APIs."""
    from src.data.pipeline import DataPipeline, PipelineConfig

    cfg = PipelineConfig(
        symbols=symbols or ["SPY", "AAPL", "MSFT"],
        min_samples=min_samples,
        output_dir="data/training",
    )
    pipeline = DataPipeline(cfg)
    await pipeline.run()
    return pipeline.load_latest_data()


async def load_or_collect_data(
    use_real_data: bool = True, n_samples: int = settings.ML_TRAINING_DEFAULT_SAMPLES
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load data with synthetic fallback."""
    if use_real_data:
        try:
            return await collect_real_data(min_samples=n_samples)
        except Exception as e:
            logger.warning("data_collection_failed", error=str(e))
    return (*generate_synthetic_data(n_samples), {"data_source": "synthetic"})


async def train(
    use_real_data: bool = True,
    n_samples: int = settings.ML_TRAINING_DEFAULT_SAMPLES,
    framework: str = "xgboost",
    params: dict[str, Any] | None = None,
    promote_threshold: float = 0.99,
) -> dict[str, Any]:
    """Execute training pipeline with MLflow tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    os.makedirs("mlruns", exist_ok=True)
    
    from src.ml.trainer import InstrumentedTrainer
    trainer = InstrumentedTrainer(
        study_name=f"Option_Pricing_{framework}",
        tracking_uri=tracking_uri
    )
    
    x_vals, y_vals, features, meta = await load_or_collect_data(use_real_data, n_samples)
    
    # Default parameters
    default_params = {
        "max_depth": settings.ML_XGBOOST_MAX_DEPTH,
        "learning_rate": settings.ML_XGBOOST_LEARNING_RATE,
        "n_estimators": settings.ML_XGBOOST_N_ESTIMATORS,
        "framework": framework
    }
    if params:
        default_params.update(params)
    
    logger.info("starting_model_training", n_samples=len(x_vals), params=default_params, meta=meta)
    
    # 🚀 SINGULARITY: Instrumented training with SOTA metrics
    accuracy = trainer.train_and_evaluate(
        x_vals, y_vals, 
        params=default_params,
        feature_names=features,
        dataset_metadata=meta
    )
    
    promoted = False
    run_id = trainer.tracker.run_id if hasattr(trainer.tracker, "run_id") else "unknown"
    
    if accuracy >= promote_threshold:
        logger.info("model_meets_promotion_threshold", accuracy=accuracy, threshold=promote_threshold)
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            model_name = f"Option_Pricing_{framework}"
            
            # Register model
            result = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name
            )
            
            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage="Production"
            )
            logger.info("model_promoted_to_production", name=model_name, version=result.version)
            promoted = True
        except Exception as e:
            logger.warning("model_promotion_failed", error=str(e))
    
    logger.info("training_complete", accuracy=accuracy, promoted=promoted, run_id=run_id)
    return {"run_id": run_id, "accuracy": accuracy, "promoted": promoted}


if __name__ == "__main__":
    asyncio.run(train())