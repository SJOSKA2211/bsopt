import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.pricing.black_scholes import BlackScholesEngine, BSParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic training data using Black-Scholes engine."""
    logger.info(f"Generating {n_samples} synthetic samples...")
    np.random.seed(42)

    S = np.random.uniform(50, 150, n_samples)
    K = np.random.uniform(50, 150, n_samples)
    T = np.random.uniform(0.1, 2.0, n_samples)
    r = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)
    is_call = np.random.choice([0, 1], n_samples)

    prices = np.zeros(n_samples)
    for i in range(n_samples):
        params = BSParameters(S[i], K[i], T[i], sigma[i], r[i], 0.0)
        prices[i] = (
            BlackScholesEngine.price_call(params)
            if is_call[i]
            else BlackScholesEngine.price_put(params)
        )

    X = np.column_stack([S, K, T, is_call, S / K, np.log(S / K), np.sqrt(T), T * 365, sigma])

    feature_names = [
        "underlying_price",
        "strike",
        "time_to_expiry",
        "is_call",
        "moneyness",
        "log_moneyness",
        "sqrt_time_to_expiry",
        "days_to_expiry",
        "implied_volatility",
    ]
    return X, prices, feature_names


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> float:
    """Optuna objective for XGBoost optimization."""
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        model = xgb.XGBRegressor(**param)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        scores.append(r2_score(y_v, model.predict(X_v)))
    return float(np.mean(scores))


def run_hyperparameter_optimization(
    use_real_data: bool = True, n_samples: int = 10000, n_trials: int = 50
) -> Dict[str, Any]:
    """Run HPO using Optuna."""
    X, y, _, _ = load_or_collect_data(use_real_data=use_real_data, n_samples=n_samples)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(trial=t, X=X_scaled, y=y), n_trials=n_trials)
    return {"best_params": study.best_params, "best_r2": study.best_value}


async def collect_real_data(
    symbols: Optional[List[str]] = None, min_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
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


def load_or_collect_data(
    use_real_data: bool = True, n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Load data with synthetic fallback."""
    if use_real_data:
        try:
            return asyncio.run(collect_real_data(min_samples=n_samples))
        except Exception as e:
            logger.warning(f"Data collection failed: {e}. Using synthetic.")
    return (*generate_synthetic_data(n_samples), {"data_source": "synthetic"})


def train(
    use_real_data: bool = True,
    n_samples: int = 10000,
    params: Optional[Dict[str, Any]] = None,
    promote_threshold: float = 0.99,
) -> Dict[str, Any]:
    """Execute training pipeline with MLflow tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_experiment("Option_Pricing_XGBoost")
    X, y, features, _ = load_or_collect_data(use_real_data, n_samples)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    params = params or {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "n_jobs": -1}
    with mlflow.start_run():
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Workaround for MLflow/XGBoost compatibility issue
        if not hasattr(model, "_estimator_type"):
            model._estimator_type = "regressor"

        r2 = r2_score(y_test, model.predict(X_test))
        mlflow.log_params(params)
        mlflow.log_metric("test_r2", r2)
        model_info = mlflow.xgboost.log_model(
            model, "model", registered_model_name="XGBoostOptionPricer"
        )

        promoted = False
        if r2 >= promote_threshold:
            MlflowClient().transition_model_version_stage(
                name="XGBoostOptionPricer",
                version=model_info.registered_model_version,
                stage="Production",
            )
            promoted = True
        return {"run_id": mlflow.active_run().info.run_id, "r2": r2, "promoted": promoted}


if __name__ == "__main__":
    train()
