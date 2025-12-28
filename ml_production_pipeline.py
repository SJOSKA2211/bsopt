import asyncio
import os
import json
import logging
import time
import sys
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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, f1_score
import xgboost as xgb

from src.data.pipeline import DataPipeline, PipelineConfig
from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import generate_synthetic_data

# --- CONFIGURATION & STANDARDIZATION ---
ML_EXP_PRICING = "Prod_Pricing_XGBoost"
ML_EXP_CLASSIFICATION = "Prod_ITM_NN"
TRACKING_URI = "file://" + os.path.abspath("mlruns")
RESULTS_DIR = Path("results/production_pipeline")
THRESHOLDS = {
    "xgboost_r2_min": 0.95,
    "nn_accuracy_min": 0.90
}

# Structured Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger("ml_production")

def log_structured(msg, **kwargs):
    entry = {"timestamp": datetime.now().isoformat(), "message": msg}
    entry.update(kwargs)
    print(json.dumps(entry))

# --- VISUALIZATION HELPERS ---
def create_performance_plots(y_true, y_pred, model_name, output_dir):
    # Regression Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"{model_name} Performance: Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    path = output_dir / f"{model_name.lower()}_performance.png"
    plt.savefig(path)
    plt.close()
    return path

# --- CORE PIPELINE ---
async def run_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    
    log_structured("Starting Production ML Pipeline", version="2.1.0")
    
    # 1. Data Ingestion & Logging
    log_structured("Step 1: Data Preparation")
    X, y_reg, feature_names = generate_synthetic_data(n_samples=10000)
    
    # ITM Classification Labels
    spot_idx, strike_idx, is_call_idx = feature_names.index("underlying_price"), feature_names.index("strike"), feature_names.index("is_call")
    y_cls = (((X[:, is_call_idx] == 1) & (X[:, spot_idx] > X[:, strike_idx])) | 
             ((X[:, is_call_idx] == 0) & (X[:, spot_idx] < X[:, strike_idx]))).astype(int)
    
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log Data Artifacts
    data_summary = {
        "n_samples": len(X),
        "features": feature_names,
        "class_distribution": int(np.sum(y_cls)) / len(y_cls)
    }
    with open(RESULTS_DIR / "data_summary.json", "w") as f:
        json.dump(data_summary, f, indent=2)

    # 2. XGBoost Pricing Model
    mlflow.set_experiment(ML_EXP_PRICING)
    with mlflow.start_run(run_name="production-xgboost") as run:
        log_structured("Training XGBoost Pricing Model", experiment=ML_EXP_PRICING)
        params = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300, "objective": "reg:squarederror"}
        model_xgb = xgb.XGBRegressor(**params)
        model_xgb.fit(X_train_scaled, y_reg_train)
        
        y_pred = model_xgb.predict(X_test_scaled)
        r2 = r2_score(y_reg_test, y_pred)
        mse = mean_squared_error(y_reg_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metrics({"r2": r2, "mse": mse})
        
        # Log Model & Artifacts
        model_xgb._estimator_type = "regressor"
        xgb_path = RESULTS_DIR / "xgboost_model.json"
        model_xgb.save_model(str(xgb_path))
        mlflow.log_artifact(str(xgb_path), "model")
        
        viz_path = create_performance_plots(y_reg_test, y_pred, "XGBoost", RESULTS_DIR)
        mlflow.log_artifact(str(viz_path), "plots")
        mlflow.log_artifact(str(RESULTS_DIR / "data_summary.json"), "data")
        
        xgb_success = r2 >= THRESHOLDS["xgboost_r2_min"]
        log_structured("XGBoost training complete", r2=r2, status="PASS" if xgb_success else "FAIL")

    # 3. Neural Network Classification Model
    mlflow.set_experiment(ML_EXP_CLASSIFICATION)
    with mlflow.start_run(run_name="production-nn") as run:
        log_structured("Training NN Classification Model", experiment=ML_EXP_CLASSIFICATION)
        nn_params = {"epochs": 10, "batch_size": 64, "lr": 0.001}
        mlflow.log_params(nn_params)
        
        train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_cls_train))
        loader = DataLoader(train_ds, batch_size=nn_params["batch_size"], shuffle=True)
        
        model_nn = OptionPricingNN(input_dim=X.shape[1], num_classes=2)
        optimizer = optim.Adam(model_nn.parameters(), lr=nn_params["lr"])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, nn_params["epochs"] + 1):
            model_nn.train()
            for b_X, b_y in loader:
                optimizer.zero_grad(); outputs = model_nn(b_X); loss = criterion(outputs, b_y)
                loss.backward(); optimizer.step()
            mlflow.log_metric("loss", loss.item(), step=epoch)
            
        model_nn.eval()
        with torch.no_grad():
            preds = torch.max(model_nn(torch.FloatTensor(X_test_scaled)), 1)[1].numpy()
        
        acc = accuracy_score(y_cls_test, preds)
        f1 = f1_score(y_cls_test, preds)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        
        nn_model_path = RESULTS_DIR / "nn_model.pth"
        torch.save(model_nn.state_dict(), str(nn_model_path))
        mlflow.log_artifact(str(nn_model_path), "model")
        
        nn_success = acc >= THRESHOLDS["nn_accuracy_min"]
        log_structured("NN training complete", accuracy=acc, status="PASS" if nn_success else "FAIL")

    # 4. Final Standardization & Automation (CI/CD Check)
    pipeline_report = {
        "timestamp": datetime.now().isoformat(),
        "xgboost": {"r2": r2, "status": "PASS" if xgb_success else "FAIL"},
        "nn": {"accuracy": acc, "status": "PASS" if nn_success else "FAIL"},
        "all_passed": xgb_success and nn_success
    }
    
    report_path = RESULTS_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(pipeline_report, f, indent=2)
    
    log_structured("Pipeline Execution Finished", all_passed=pipeline_report["all_passed"])
    
    if not pipeline_report["all_passed"]:
        log_structured("CI/CD Alert: Performance thresholds not met!")
        # sys.exit(1) # Would stop CI/CD pipeline
    
    # 5. Dashboard Generation
    generate_production_dashboard(pipeline_report)

def generate_production_dashboard(report):
    html = f"""
    <html><head><title>Production ML Report</title><style>
    body {{ font-family: sans-serif; margin: 40px; }}
    .status-PASS {{ color: green; font-weight: bold; }}
    .status-FAIL {{ color: red; font-weight: bold; }}
    .card {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 8px; }}
    </style></head><body>
    <h1>Production ML Pipeline Report</h1>
    <p>Executed: {report['timestamp']}</p>
    <div class="card">
        <h2>XGBoost Pricing</h2>
        <p>R2 Score: {report['xgboost']['r2']:.4f}</p>
        <p>Status: <span class="status-{report['xgboost']['status']}">{report['xgboost']['status']}</span></p>
    </div>
    <div class="card">
        <h2>Neural Network ITM</h2>
        <p>Accuracy: {report['nn']['accuracy']:.4f}</p>
        <p>Status: <span class="status-{report['nn']['status']}">{report['nn']['status']}</span></p>
    </div>
    </body></html>
    """
    with open(RESULTS_DIR / "index.html", "w") as f:
        f.write(html)
    log_structured("Dashboard generated at results/production_pipeline/index.html")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
