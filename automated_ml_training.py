import click
import json
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score
import xgboost as xgb

from src.ml.training.train import load_or_collect_data
from src.ml.architectures.neural_network import OptionPricingNN

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@click.command()
@click.option('--dataset', type=click.Path(exists=True), required=True, help='Path to the dataset CSV')
@click.option('--model', type=click.Choice(['xgboost', 'nn']), required=True, help='Model type: xgboost or nn')
@click.option('--params', type=click.Path(exists=True), help='Path to JSON file containing hyperparameters')
@click.option('--output', type=click.Path(), required=True, help='Directory to save the trained model')
@click.option('--eval', 'eval_metrics', required=True, help='Comma-separated metrics to evaluate (e.g., mse,r2,accuracy)')
@click.option('--epochs', type=int, default=10, help='Number of training epochs (for NN)')
@click.option('--batch_size', type=int, default=32, help='Batch size for training (for NN)')
def main(dataset, model, params, output, eval_metrics, epochs, batch_size):
    """
    Automated ML training pipeline for option pricing.
    """
    os.makedirs(output, exist_ok=True)
    
    # Set tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    
    # 1. Load Data
    logger.info(f"Loading dataset from {dataset}...")
    df = pd.read_csv(dataset)
    
    # Basic validation and feature preparation
    # Assuming standard format: underlying_price, strike, time_to_expiry, risk_free_rate, volatility, is_call, price
    required_cols = ['underlying_price', 'strike', 'time_to_expiry', 'risk_free_rate', 'volatility', 'is_call', 'price']
    for col in required_cols:
        if col not in df.columns:
            # Try to map columns if they have slightly different names
            mapping = {
                'spot': 'underlying_price',
                'rate': 'risk_free_rate',
                'sigma': 'volatility',
                'target': 'price'
            }
            if col in mapping.values():
                # reverse lookup
                orig = [k for k, v in mapping.items() if v == col][0]
                if orig in df.columns:
                    df[col] = df[orig]
                else:
                    logger.error(f"Missing required column: {col}")
                    return

    X = df[['underlying_price', 'strike', 'time_to_expiry', 'risk_free_rate', 'volatility', 'is_call']].values
    y = df['price'].values
    
    # 2. Load Hyperparameters
    hyperparams = {}
    if params:
        with open(params, 'r') as f:
            hyperparams = json.load(f)
    
    # 3. Train Model
    mlflow.set_experiment(f"Automated_{model.upper()}")
    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(hyperparams)
        mlflow.log_param("model_type", model)
        mlflow.log_param("dataset", dataset)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model == 'xgboost':
            logger.info("Training XGBoost model...")
            xgb_params = {
                "objective": "reg:squarederror",
                "max_depth": hyperparams.get("max_depth", 6),
                "learning_rate": hyperparams.get("learning_rate", 0.1),
                "n_estimators": hyperparams.get("n_estimators", 100),
                "n_jobs": -1
            }
            model_obj = xgb.XGBRegressor(**xgb_params)
            model_obj.fit(X_train_scaled, y_train)
            
            y_pred = model_obj.predict(X_test_scaled)
            model_obj._estimator_type = "regressor"
            model_path = os.path.join(output, "model.json")
            model_obj.save_model(model_path)
            mlflow.log_artifact(model_path, "model")
            
        elif model == 'nn':
            logger.info("Training Neural Network model...")
            input_dim = X.shape[1]
            model_obj = OptionPricingNN(input_dim=input_dim)
            
            # Use regression setup as default for price prediction
            train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train).view(-1, 1))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = optim.Adam(model_obj.parameters(), lr=hyperparams.get("lr", 0.001))
            criterion = nn.MSELoss()
            
            model_obj.train()
            for epoch in range(1, epochs + 1):
                total_loss = 0
                for b_X, b_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model_obj(b_X)
                    loss = criterion(outputs, b_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
                mlflow.log_metric("loss", avg_loss, step=epoch)
            
            model_obj.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                y_pred = model_obj(X_test_tensor).numpy().flatten()
            
            model_path = os.path.join(output, "model.pth")
            torch.save(model_obj.state_dict(), model_path)
            mlflow.log_artifact(model_path, "model")
            
        # 4. Evaluation
        metrics_to_calc = [m.strip().lower() for m in eval_metrics.split(',')]
        results = {}
        
        logger.info(f"Evaluating metrics: {metrics_to_calc}")
        for m in metrics_to_calc:
            if m == 'mse':
                val = mean_squared_error(y_test, y_pred)
            elif m == 'mae':
                val = mean_absolute_error(y_test, y_pred)
            elif m == 'r2':
                val = r2_score(y_test, y_pred)
            elif m == 'accuracy':
                # Mock accuracy for regression by checking if within 5%
                val = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9)) < 0.05)
            else:
                logger.warning(f"Unknown metric: {m}")
                continue
            
            results[m] = float(val)
            mlflow.log_metric(m, val)
            logger.info(f"Metric {m}: {val:.4f}")
            
        # Save results to output dir
        with open(os.path.join(output, "evaluation.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training completed successfully. Results saved to {output}")

if __name__ == '__main__':
    main()
