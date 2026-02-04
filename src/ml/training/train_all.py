"""
Master Training Pipeline
========================

Orchestrates data collection from multiple sources (yfinance, NSE)
and trains all ML models with specified hyperparameters.

Models:
1. XGBoost Regressor (Option Pricing)
2. Neural Network Classifier (ITM/OTM Classification)
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.pipeline import DataPipeline, PipelineConfig
from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import train as train_xgboost
from src.ml.training.train_v2 import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    prepare_classification_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Collect fresh dataset from yfinance and NSE."""
    logger.info("Starting data collection from yfinance and NSE...")

    config = PipelineConfig(
        symbols=[
            "SPY",
            "AAPL",
            "MSFT",
            "NVDA",  # yfinance
            "NIFTY",
            "BANKNIFTY",
            "RELIANCE",  # NSE
        ],
        min_samples=20000,
        use_multi_source=True,
        validate_data=True,
        output_dir="data/training",
    )

    pipeline = DataPipeline(config)
    report = await pipeline.run()

    logger.info(f"Data collection report: {report}")
    return pipeline.load_latest_data()


def train_all():
    """Train all models using the collected dataset."""
    # 1. Collect Data
    try:
        X, y_reg, feature_names, metadata = asyncio.run(collect_dataset())
    except Exception as e:
        logger.error(f"Failed to collect real data: {e}. Falling back to synthetic.")
        from src.ml.training.train import generate_synthetic_data

        X, y_reg, feature_names = generate_synthetic_data(10000)
        metadata = {"source": "synthetic"}

    # 2. Train XGBoost Regressor
    logger.info("Training XGBoost Regressor...")
    xgb_results = train_xgboost(use_real_data=True, n_samples=len(X))
    logger.info(f"XGBoost Training Results: {xgb_results}")

    # 3. Train NN Classifier
    logger.info("Training NN Classifier...")
    # Prepare classification targets
    y_class = prepare_classification_data(X, feature_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 🚀 SINGULARITY: Strict Temporal Split (No Shuffling)
    # Market data must be validated sequentially to avoid leakage.
    test_size = 0.2
    split_idx = int(len(X_scaled) * (1 - test_size))
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    logger.info("temporal_split_complete", train_len=len(X_train), test_len=len(X_test))

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = OptionPricingNN(input_dim=len(feature_names), num_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 4. MLflow Tracking
    from src.config import get_settings
    settings = get_settings()
    
    # MLflow needs standard postgresql prefix (not asyncpg)
    tracking_uri = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info("mlflow_tracking_redirected", target="neon")

    mlflow.set_experiment("Option_ITM_Classification")
    with mlflow.start_run(run_name=f"master-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "optimizer": "AdamW",
                "source": metadata.get("source", "real"),
            }
        )

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_acc = val_correct / val_total
            logger.info(f"Epoch {epoch}/{EPOCHS} - Val Acc: {val_acc:.4f}")
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)

        mlflow.pytorch.log_model(model, "itm_classifier")
        logger.info("NN Classifier training complete and logged to MLflow.")


if __name__ == "__main__":
    train_all()
