"""
Cross-Sectional Machine Learning Pipeline
=========================================
Extracts populated dataset from PostgreSQL, performs feature engineering
(calculating Black-Scholes outputs), cleans data, and trains a PyTorch
model cross-sectionally on the entire universe of stocks.
"""

import argparse
from typing import Any, cast

import mlflow
import ray
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader, TensorDataset

from src.config import settings
from src.shared.math_utils import calculate_greeks, calculate_price

logger = structlog.get_logger(__name__)

# ─── Data Extraction & Feature Engineering ────────────────────────────────


def extract_and_engineer_features(chunk_size: int = 50000) -> pd.DataFrame:
    """
    Extracts options data in chunks from PostgreSQL to avoid OOM.
    Performs dynamic feature engineering using vectorized Black-Scholes.
    """
    logger.info("ml_extraction_start")

    # We use sync SQLAlchemy engine for chunked Pandas read
    engine = create_engine(settings.DATABASE_URL)

    query = """
        SELECT 
            o.time as timestamp, o.symbol, o.strike, o.expiry, o.option_type,
            o.last as market_price, o.volume, o.open_interest, o.implied_volatility,
            t.price as spot,
            0.05 as risk_free_rate,
            0.0 as dividend_yield
        FROM options_prices o
        JOIN LATERAL (
            SELECT price 
            FROM market_ticks mt 
            WHERE mt.symbol = o.symbol AND mt.time <= o.time
            ORDER BY mt.time DESC 
            LIMIT 1
        ) t ON TRUE
        WHERE o.time >= '2020-01-01'
        ORDER BY o.time ASC
    """

    chunks: list[pd.DataFrame] = []
    with engine.connect() as conn:
        for chunk in pd.read_sql(text(query), conn, chunksize=chunk_size):
            # 1. Rigorous Data Cleaning
            chunk = chunk.dropna(subset=["symbol", "strike", "expiry", "option_type"])
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
            chunk["expiry"] = pd.to_datetime(chunk["expiry"], utc=True)
            chunk["is_halted"] = chunk["volume"] == 0
            chunk = chunk.sort_values(["symbol", "timestamp"])

            chunk["market_price"] = chunk.groupby("symbol")["market_price"].transform(
                lambda x: x.interpolate(method="linear", limit=5).ffill().bfill()
            )
            chunk = chunk.dropna(subset=["market_price"])
            chunk["T"] = (chunk["expiry"] - chunk["timestamp"]).dt.total_seconds() / (
                365.25 * 24 * 3600
            )

            chunk = chunk[
                (chunk["T"] > (1.0 / 365.25))
                & (chunk["market_price"] > 0.01)
                & (chunk["implied_volatility"].fillna(0.2).between(0.01, 5.0))
            ]
            chunk["implied_volatility"] = chunk["implied_volatility"].fillna(0.2)

            if chunk.empty:
                continue

            # 2. Vectorized Feature Engineering (Black-Scholes Greeks)
            s = cast(np.ndarray[Any, np.dtype[np.float64]], chunk["spot"].values)
            k = cast(np.ndarray[Any, np.dtype[np.float64]], chunk["strike"].values)
            t = cast(np.ndarray[Any, np.dtype[np.float64]], chunk["T"].values)
            sigma = cast(np.ndarray[Any, np.dtype[np.float64]], chunk["implied_volatility"].values)
            r = cast(np.ndarray[Any, np.dtype[np.float64]], chunk["risk_free_rate"].values)
            q = cast(np.ndarray[Any, np.dtype[np.float64]], chunk["dividend_yield"].values)
            is_call = cast(
                np.ndarray[Any, np.dtype[np.bool_]], (chunk["option_type"] == "call").values
            )

            # Calculate Greeks using our highly optimized pure NumPy math utilities
            delta, gamma, theta, vega, rho = calculate_greeks(s, k, t, sigma, r, q, is_call)

            chunk["bs_delta"] = delta
            chunk["bs_gamma"] = gamma
            chunk["bs_theta"] = theta
            chunk["bs_vega"] = vega
            chunk["bs_rho"] = rho
            chunk["bs_price"] = calculate_price(s, k, t, sigma, r, q, is_call)

            # Additional engineered features for cross-sectional learning
            chunk["moneyness"] = s / k
            chunk["log_moneyness"] = np.log(np.maximum(s / k, 1e-8))
            chunk["sqrt_T"] = np.sqrt(np.maximum(t, 1e-8))
            chunk["vega_gamma_ratio"] = np.where(np.abs(gamma) > 1e-12, vega / (gamma + 1e-12), 0.0)

            chunks.append(chunk)
            logger.info("ml_extraction_chunk_processed", size=len(chunk))

    if not chunks:
        return pd.DataFrame()

    final_df = pd.concat(chunks, ignore_index=True)
    logger.info("ml_extraction_complete", total_rows=len(final_df))
    return final_df


# ─── PyTorch Model ──────────────────────────────────────────────────────────


class CrossSectionalPricingModel(nn.Module):  # type: ignore
    """Deep Neural Network for cross-sectional option pricing."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Predicts Price
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


# ─── Training Loop ──────────────────────────────────────────────────────────


def train_pipeline(
    epochs: int = 10,
    batch_size: int = 1024,
    study_name: str = "cross_sectional_v1",
    tracking_uri: str = "http://mlflow:5000",
) -> None:
    """
    End-to-end Machine Learning Pipeline for Cross-Sectional Option Pricing.

    Workflow:
    1. Extract data from TimescaleDB with high-fidelity spot price joins.
    2. Engineer vectorized features using Numba-accelerated math kernels.
    3. Perform temporal validation split (pre/post 2025).
    4. Train Deep Neural Network with AdamW and Batch Normalization.
    5. Log all telemetry, artifacts, and champion metrics to MLflow.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(study_name)
    mlflow.autolog()
    
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    with mlflow.start_run() as run:
        logger.info("ml_pipeline_ignition", run_id=run.info.run_id)

        # --- Data Acquisition ---
        df = extract_and_engineer_features()
        if df.empty:
            logger.error("ml_pipeline_aborted_no_data")
            return

        # Select production feature set
        features = [
            "spot",
            "strike",
            "T",
            "implied_volatility",
            "risk_free_rate",
            "bs_delta",
            "bs_gamma",
            "bs_vega",
            "bs_price",
            "moneyness",
            "log_moneyness",
            "sqrt_T",
        ]
        target = "market_price"

        # --- Temporal Data Partitioning ---
        # We split by date to ensure the model generalizes to future market regimes.
        split_date = pd.to_datetime("2025-01-01", utc=True)
        train_df = df[df["timestamp"] < split_date].copy()
        test_df = df[df["timestamp"] >= split_date].copy()

        if train_df.empty or test_df.empty:
            logger.warning("ml_pipeline_insufficient_temporal_data", split_date=str(split_date))
            # Fallback to simple ratio split if temporal data is sparse
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

        # --- Feature Preparation & Normalization (Optimized) ---
        from src.ml.pre_training import MLPreTrainer

        # Cross-sectional feature enrichment
        df = MLPreTrainer.calculate_cross_sectional_features(df)

        # select potentially enriched features
        features = [f for f in features if f in df.columns]

        X_all, means, stds = MLPreTrainer.prepare_features(df, features)
        y_all = df[target].values.astype(np.float64)

        # Log normalization constants for inference parity
        mlflow.log_dict({"means": means.tolist(), "features": features}, "normalization/means.json")
        mlflow.log_dict({"stds": stds.tolist(), "features": features}, "normalization/stds.json")

        # Temporal split on raw arrays
        split_idx = len(df[df["timestamp"] < split_date])
        if split_idx == 0 or split_idx == len(X_all):
            split_idx = int(len(X_all) * 0.8)

        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        # --- Model Preparation ---
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        model = CrossSectionalPricingModel(input_dim=len(features))
        
        # HIGH-PERFORMANCE: Use torch.compile for graph optimization if available
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                logger.info("pytorch_model_compiled")
            except Exception as e:
                logger.warning("pytorch_compile_failed", error=str(e))

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": "AdamW",
                "lr_scheduler": "CosineAnnealingLR",
                "weight_decay": 1e-4,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(features),
            }
        )

        # --- Training Execution with Early Stopping ---
        logger.info("ml_training_started", epochs=epochs, train_samples=len(X_train))
        best_test_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += float(loss.item()) * batch_X.size(0)

            scheduler.step()
            epoch_loss = running_loss / len(train_dataset)

            # --- Continuous Evaluation ---
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                test_loss = float(criterion(test_preds, y_test_t).item())

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
            logger.info(
                "ml_epoch_metrics", epoch=epoch + 1, train_loss=epoch_loss, test_loss=test_loss
            )

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("early_stopping_triggered", epoch=epoch + 1)
                    break

        # --- Finalization & Promotion ---
        logger.info("ml_training_complete")

        # Log requirements for reproducibility
        mlflow.set_tag("framework", "pytorch")
        mlflow.pytorch.log_model(model, "model")

        # Calculate final R2 Score
        from sklearn.metrics import mean_absolute_error, r2_score

        final_preds = model(X_test_t).detach().numpy()
        r2 = r2_score(y_test, final_preds)
        mae = mean_absolute_error(y_test, final_preds)
        rmse = float(np.sqrt(test_loss))
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        logger.info("ml_final_evaluation", r2_score=r2, mae=mae, rmse=rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--study_name", type=str, default="cross_sectional_v1")
    parser.add_argument("--tracking_uri", type=str, default="http://mlflow:5000")
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        study_name=args.study_name,
        tracking_uri=args.tracking_uri,
    )
