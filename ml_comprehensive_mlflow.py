import asyncio
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.xgboost
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, confusion_matrix
import xgboost as xgb

from src.data.pipeline import DataPipeline, PipelineConfig
from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import generate_synthetic_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_regression_results(y_true, y_pred, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

async def run_comprehensive_mlflow():
    """
    Comprehensive MLflow run for all models.
    """
    output_base = Path("results/comprehensive_run")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Set tracking URI - preferring local file for reliability in this environment
    tracking_uri = "file://" + os.path.abspath("mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    
    logger.info("Step 1: Data Collection & Preparation")
    # For speed in this demonstration, we'll use a mix of synthetic and real data logic
    X, y_reg, feature_names = generate_synthetic_data(n_samples=5000)
    
    # Prepare classification data (ITM/OTM)
    spot_idx = feature_names.index("underlying_price")
    strike_idx = feature_names.index("strike")
    is_call_idx = feature_names.index("is_call")
    itm_labels = (( (X[:, is_call_idx] == 1) & (X[:, spot_idx] > X[:, strike_idx]) ) | 
                  ( (X[:, is_call_idx] == 0) & (X[:, spot_idx] < X[:, strike_idx]) )).astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, itm_labels, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------------------
    # XGBoost: Option Pricing (Regression)
    # -------------------------------------------------------------------------
    mlflow.set_experiment("Comprehensive_XGBoost_Pricing")
    with mlflow.start_run(run_name=f"xgboost-pricing-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        logger.info("Training XGBoost Regressor...")
        xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "n_jobs": -1
        }
        mlflow.log_params(xgb_params)
        mlflow.log_param("n_samples", len(X_train))
        
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train_scaled, y_reg_train)
        
        y_pred_xgb = model_xgb.predict(X_test_scaled)
        mse = mean_squared_error(y_reg_test, y_pred_xgb)
        r2 = r2_score(y_reg_test, y_pred_xgb)
        
        mlflow.log_metrics({"mse": mse, "r2": r2})
        
        # Save model workaround for estimator type
        model_xgb._estimator_type = "regressor"
        xgb_local_path = output_base / "xgb_model.json"
        model_xgb.save_model(str(xgb_local_path))
        mlflow.log_artifact(str(xgb_local_path), "model")
        
        # Visualize
        plot_path = output_base / "xgb_regression_plot.png"
        plot_regression_results(y_reg_test, y_pred_xgb, "XGBoost Pricing Performance", str(plot_path))
        mlflow.log_artifact(str(plot_path), "visualizations")
        
        logger.info(f"XGBoost run completed. R2: {r2:.4f}")

    # -------------------------------------------------------------------------
    # Neural Network: ITM Classification
    # -------------------------------------------------------------------------
    mlflow.set_experiment("Comprehensive_NN_Classification")
    with mlflow.start_run(run_name=f"nn-classification-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        logger.info("Training NN Classifier...")
        
        nn_config = {
            "epochs": 5,
            "batch_size": 32,
            "lr": 0.001,
            "hidden_dim": 64
        }
        mlflow.log_params(nn_config)
        
        # Prepare datasets
        train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_cls_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_cls_test))
        train_loader = DataLoader(train_ds, batch_size=nn_config["batch_size"], shuffle=True)
        
        model_nn = OptionPricingNN(input_dim=X.shape[1], num_classes=2)
        optimizer = optim.Adam(model_nn.parameters(), lr=nn_config["lr"])
        criterion = nn.CrossEntropyLoss()
        
        model_nn.train()
        for epoch in range(1, nn_config["epochs"] + 1):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model_nn(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            logger.info(f"NN Epoch {epoch}: loss={avg_loss:.4f}")
            
        model_nn.eval()
        with torch.no_grad():
            test_outputs = model_nn(torch.FloatTensor(X_test_scaled))
            _, y_pred_nn = torch.max(test_outputs, 1)
            y_pred_nn = y_pred_nn.numpy()
            
        acc = accuracy_score(y_cls_test, y_pred_nn)
        prec = precision_score(y_cls_test, y_pred_nn)
        
        mlflow.log_metrics({"accuracy": acc, "precision": prec})
        
        # Save model
        nn_local_path = output_base / "nn_model.pth"
        torch.save(model_nn.state_dict(), str(nn_local_path))
        mlflow.log_artifact(str(nn_local_path), "model")
        
        # Visualize
        cm_path = output_base / "nn_confusion_matrix.png"
        plot_confusion_matrix(y_cls_test, y_pred_nn, "NN ITM Classification Performance", str(cm_path))
        mlflow.log_artifact(str(cm_path), "visualizations")
        
        logger.info(f"NN run completed. Accuracy: {acc:.4f}")

    # Final summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "xgboost": {"r2": float(r2), "mse": float(mse)},
        "neural_network": {"accuracy": float(acc), "precision": float(prec)},
        "mlflow_tracking_uri": tracking_uri,
        "status": "success"
    }
    
    summary_path = output_base / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Comprehensive MLflow run completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_mlflow())
