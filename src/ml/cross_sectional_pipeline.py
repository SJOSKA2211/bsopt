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
            time as timestamp, symbol, strike, expiry, option_type,
            last as market_price, volume, open_interest, implied_volatility,
            100.0 as spot, -- Stub: Needs to join with market_ticks for true spot
            0.05 as risk_free_rate,
            0.0 as dividend_yield
        FROM options_prices
        WHERE time >= '2020-01-01'
        ORDER BY time ASC
    """
    
    chunks: list[pd.DataFrame] = []
    with engine.connect() as conn:
        for chunk in pd.read_sql(text(query), conn, chunksize=chunk_size):
            # 1. Rigorous Data Cleaning
            chunk = chunk.dropna(subset=['symbol', 'strike', 'expiry', 'option_type'])
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
            chunk['expiry'] = pd.to_datetime(chunk['expiry'], utc=True)
            chunk['is_halted'] = chunk['volume'] == 0
            chunk = chunk.sort_values(['symbol', 'timestamp'])

            chunk['market_price'] = chunk.groupby('symbol')['market_price'].transform(
                lambda x: x.interpolate(method='linear', limit=5).ffill().bfill()
            )
            chunk = chunk.dropna(subset=['market_price'])
            chunk['T'] = (chunk['expiry'] - chunk['timestamp']).dt.total_seconds() / (365.25 * 24 * 3600)

            chunk = chunk[
                (chunk['T'] > (1.0 / 365.25)) & 
                (chunk['market_price'] > 0.01) &
                (chunk['implied_volatility'].fillna(0.2).between(0.01, 5.0))
            ]
            chunk['implied_volatility'] = chunk['implied_volatility'].fillna(0.2)

            if chunk.empty:
                continue

            # 2. Vectorized Feature Engineering (Black-Scholes Greeks)
            s = cast(np.ndarray[Any, np.dtype[np.float64]], chunk['spot'].values)
            k = cast(np.ndarray[Any, np.dtype[np.float64]], chunk['strike'].values)
            t = cast(np.ndarray[Any, np.dtype[np.float64]], chunk['T'].values)
            sigma = cast(np.ndarray[Any, np.dtype[np.float64]], chunk['implied_volatility'].values)
            r = cast(np.ndarray[Any, np.dtype[np.float64]], chunk['risk_free_rate'].values)
            q = cast(np.ndarray[Any, np.dtype[np.float64]], chunk['dividend_yield'].values)
            is_call = cast(np.ndarray[Any, np.dtype[np.bool_]], (chunk['option_type'] == 'call').values)

            # Calculate Greeks using our highly optimized pure NumPy math utilities
            delta, gamma, theta, vega, rho = calculate_greeks(s, k, t, sigma, r, q, is_call)
            
            chunk['bs_delta'] = delta
            chunk['bs_gamma'] = gamma
            chunk['bs_theta'] = theta
            chunk['bs_vega'] = vega
            chunk['bs_rho'] = rho
            chunk['bs_price'] = calculate_price(s, k, t, sigma, r, q, is_call)
            
            chunks.append(chunk)
            logger.info("ml_extraction_chunk_processed", size=len(chunk))

    if not chunks:
        return pd.DataFrame()
        
    final_df = pd.concat(chunks, ignore_index=True)
    logger.info("ml_extraction_complete", total_rows=len(final_df))
    return final_df

# ─── PyTorch Model ──────────────────────────────────────────────────────────

class CrossSectionalPricingModel(nn.Module): # type: ignore
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
            nn.Linear(hidden_dim // 2, 1)  # Predicts Price
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))

# ─── Training Loop ──────────────────────────────────────────────────────────

def train_pipeline(epochs: int = 10, batch_size: int = 1024, study_name: str = "cross_sectional_v1", tracking_uri: str = "http://mlflow:5000") -> None:
    """End-to-end ML Pipeline."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(study_name)
    
    with mlflow.start_run():
        df = extract_and_engineer_features()
        if df.empty:
            logger.warning("ml_pipeline_no_data")
            return

        # Select features
        features = ['spot', 'strike', 'T', 'implied_volatility', 'risk_free_rate', 
                    'bs_delta', 'bs_gamma', 'bs_vega', 'bs_price']
        target = 'market_price'
        
        # 3. Temporal Train/Test Split
        split_date = pd.to_datetime('2025-01-01', utc=True)
        train_df = df[df['timestamp'] < split_date].copy()
        test_df = df[df['timestamp'] >= split_date].copy()
        
        if train_df.empty or test_df.empty:
            logger.warning("ml_pipeline_insufficient_data_for_split")
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        
        # Normalize Features
        feature_means = train_df[features].mean()
        feature_stds = train_df[features].std().replace(0, 1)
        
        X_train = cast(np.ndarray[Any, np.dtype[np.float64]], ((train_df[features] - feature_means) / feature_stds).values)
        y_train = cast(np.ndarray[Any, np.dtype[np.float64]], train_df[target].values)
        X_test = cast(np.ndarray[Any, np.dtype[np.float64]], ((test_df[features] - feature_means) / feature_stds).values)
        y_test = cast(np.ndarray[Any, np.dtype[np.float64]], test_df[target].values)

        # Convert to PyTorch Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize Model
        model = CrossSectionalPricingModel(input_dim=len(features))
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Log params
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        })

        # Training Loop
        logger.info("ml_training_started", epochs=epochs, train_samples=len(X_train))
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * batch_X.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                test_loss = float(criterion(test_preds, y_test_t).item())
                
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            logger.info("ml_epoch_metrics", epoch=epoch+1, train_loss=epoch_loss, test_loss=test_loss)
            
        logger.info("ml_training_complete")
        mlflow.pytorch.log_model(model, "model")

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
        tracking_uri=args.tracking_uri
    )
