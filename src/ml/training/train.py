from typing import Any, Dict, List, Optional, Tuple, cast
import urllib.parse
import re
import logging # Add this line

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.pricing.black_scholes import BlackScholesEngine, BSParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _validate_mlflow_tracking_uri(uri: str) -> str:
    """
    Validates the MLFLOW_TRACKING_URI to prevent SSRF and LFI.
    Allowed schemes: 'file' (only for specific local paths), 'http', 'https', 'databricks'.
    Allowed hosts for 'http/https': specific MLflow server URLs.
    Allowed local paths for 'file': within the project's 'mlruns' directory.
    """
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI cannot be empty.")

    parsed_uri = urllib.parse.urlparse(uri)

    # Allowlist of schemes
    allowed_schemes = ["http", "https", "databricks", "file"]
    if parsed_uri.scheme not in allowed_schemes:
        raise ValueError(f"MLFLOW_TRACKING_URI scheme '{parsed_uri.scheme}' not allowed. Must be one of {allowed_schemes}.")

    if parsed_uri.scheme in ["http", "https"]:
        # For production, replace with your actual MLflow server hostnames
        allowed_mlflow_hosts = [
            "localhost",
            "127.0.0.1",
            "mlflow-server.internal.com", # Example internal MLflow server
            "mlflow.example.com",         # Example external MLflow server
        ]
        if parsed_uri.hostname not in allowed_mlflow_hosts:
            raise ValueError(f"MLFLOW_TRACKING_URI host '{parsed_uri.hostname}' not allowed for scheme '{parsed_uri.scheme}'.")
        # Basic port validation
        if parsed_uri.port and not (1 <= parsed_uri.port <= 65535):
            raise ValueError(f"Invalid port in MLFLOW_TRACKING_URI: {parsed_uri.port}")
            
    elif parsed_uri.scheme == "file":
        # Only allow file URIs within the designated 'mlruns' directory
        abs_path = os.path.abspath(parsed_uri.path)
        mlruns_dir = os.path.abspath("mlruns")
        if not abs_path.startswith(mlruns_dir):
            raise ValueError(f"File URI '{uri}' is not allowed outside the '{mlruns_dir}' directory.")
        # Ensure it points to a directory, not an arbitrary file
        if not (os.path.isdir(abs_path) or not os.path.exists(abs_path)): # Allow creating new directories within mlruns
            raise ValueError(f"File URI '{uri}' must point to a directory within '{mlruns_dir}'.")
            
    # For 'databricks' scheme, additional validation might be needed based on specific Databricks integration
    
    return uri

def generate_synthetic_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic training data using Black-Scholes engine."""
    logger.info(f"Generating {n_samples} synthetic samples...")
    np.random.seed(settings.ML_TRAINING_RANDOM_STATE) # Use setting for random state

    S = np.random.uniform(50, 150, n_samples)
    K = np.random.uniform(50, 150, n_samples)
    T = np.random.uniform(0.1, 2.0, n_samples)
    r = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)
    is_call = np.random.choice([0, 1], n_samples)

    # Vectorize Black-Scholes calculations
    prices = cast(np.ndarray, BlackScholesEngine.price_options(
        spot=S,
        strike=K,
        maturity=T,
        volatility=sigma,
        rate=r,
        dividend=np.zeros(n_samples), # All dividends are 0.0
        option_type=np.where(is_call == 1, "call", "put") # Convert boolean to "call" or "put" strings
    ))

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
        "random_state": settings.ML_TRAINING_RANDOM_STATE, # Use setting for random state
        "n_jobs": -1,
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=settings.ML_TRAINING_RANDOM_STATE) # Use setting for random state
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        model = xgb.XGBRegressor(**param)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        scores.append(r2_score(y_v, model.predict(X_v)))
    return float(np.mean(scores))


async def run_hyperparameter_optimization(
    use_real_data: bool = True, n_samples: int = 10000, n_trials: Optional[int] = None
) -> Dict[str, Any]:
    """Run HPO using Optuna."""
    if n_trials is None:
        n_trials = settings.ML_TRAINING_OPTUNA_TRIALS
    X, y, _, _ = await load_or_collect_data(use_real_data=use_real_data, n_samples=n_samples)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(trial=t, X=X_scaled, y=y, n_folds=settings.ML_TRAINING_KFOLD_SPLITS), n_trials=n_trials)
    return {"best_params": study.best_params, "best_r2": study.best_value}


_ALLOWED_SYMBOLS = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"] # Example allowlist
_MAX_SYMBOLS = 10 # Example limit

async def collect_real_data(
    symbols: Optional[List[str]] = None, min_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Collect options data from market APIs."""
    from src.data.pipeline import DataPipeline, PipelineConfig

    # --- SECURITY: Symbol Validation and Limiting ---
    if symbols:
        if len(symbols) > _MAX_SYMBOLS:
            raise ValueError(f"Too many symbols requested. Maximum allowed is {_MAX_SYMBOLS}.")
        
        for symbol in symbols:
            if symbol not in _ALLOWED_SYMBOLS:
                raise ValueError(f"Symbol '{symbol}' is not in the allowed list.")
    else:
        symbols = ["SPY", "AAPL", "MSFT"] # Default if none provided and no allowlist check is done yet

    cfg = PipelineConfig(
        symbols=symbols,
        min_samples=min_samples,
        output_dir="data/training",
    )
    pipeline = DataPipeline(cfg)
    await pipeline.run()
    return pipeline.load_latest_data()


async def load_or_collect_data( # Changed to async
    use_real_data: bool = True, n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Load data with synthetic fallback."""
    if use_real_data:
        try:
            return await collect_real_data(min_samples=n_samples) # Changed to await
        except Exception as e:
            logger.warning(f"Data collection failed: {e}. Using synthetic.")
    return (*generate_synthetic_data(n_samples), {"data_source": "synthetic"})


async def train( # Changed to async
    use_real_data: bool = True,
    n_samples: int = 10000,
    params: Optional[Dict[str, Any]] = None,
    promote_threshold: float = 0.99,
) -> Dict[str, Any]:
    """Execute training pipeline with MLflow tracking."""
    raw_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
    try:
        tracking_uri = _validate_mlflow_tracking_uri(raw_tracking_uri)
    except ValueError as e:
        logger.critical(f"SSRF/LFI Prevention: Invalid MLFLOW_TRACKING_URI configured. Shutting down. Error: {e}")
        # In a real application, you might want to log this and exit
        raise

    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_experiment("Option_Pricing_XGBoost")
    X, y, features, _ = await load_or_collect_data(use_real_data, n_samples) # Changed to await
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=settings.ML_TRAINING_TEST_SIZE, random_state=settings.ML_TRAINING_RANDOM_STATE)

    params = params or {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "n_jobs": -1}
    with mlflow.start_run():
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # MLflow should infer the model type from xgb.XGBRegressor directly.
        # If issues arise, a model_signature should be explicitly defined during logging.
        r2 = r2_score(y_test, model.predict(X_test))
        mlflow.log_params(params)
        mlflow.log_metric("test_r2", r2)
        model_info = mlflow.xgboost.log_model(
            model, "model", registered_model_name="XGBoostOptionPricer"
        )

        promoted = False
        if r2 >= settings.ML_TRAINING_PROMOTE_THRESHOLD_R2: # Use setting for promotion threshold
            MlflowClient().transition_model_version_stage(
                name="XGBoostOptionPricer",
                version=model_info.registered_model_version,
                stage="Production",
            )
            promoted = True
        
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else "unknown"
        return {"run_id": run_id, "r2": r2, "promoted": promoted}


if __name__ == "__main__":
    asyncio.run(train()) # Run async train function
