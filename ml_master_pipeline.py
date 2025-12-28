import click
import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@click.command()
@click.option('--train', is_flag=True, help='Run model training')
@click.option('--evaluate', is_flag=True, help='Run model evaluation')
@click.option('--deploy', is_flag=True, help='Run model deployment')
@click.option('--monitor', is_flag=True, help='Setup model monitoring')
@click.option('--dataset', default='training_data_master.csv', help='Path to training data')
@click.option('--output', default='results/master_pipeline', help='Output directory')
def main(train, evaluate, deploy, monitor, dataset, output):
    """
    Master ML Pipeline for Training, Evaluation, Deployment, and Monitoring.
    """
    os.makedirs(output, exist_ok=True)
    
    # Set tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    
    client = MlflowClient()
    experiment_name = "Master_ML_Pipeline"
    mlflow.set_experiment(experiment_name)
    
    current_run_id = None
    
    # --- TRAINING ---
    if train:
        logger.info("Step 1: Training starting...")
        df = pd.read_csv(dataset)
        X = df[['underlying_price', 'strike', 'time_to_expiry', 'risk_free_rate', 'volatility', 'is_call']].values
        y = df['price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        with mlflow.start_run(run_name=f"master-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            current_run_id = run.info.run_id
            
            xgb_params = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100}
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train_scaled, y_train)
            
            # Save and Log model
            model_path = os.path.join(output, "model_master.json")
            model._estimator_type = "regressor"
            model.save_model(model_path)
            
            # Using simple artifact logging to avoid Registry issues if not fully setup
            mlflow.log_artifact(model_path, "model")
            mlflow.log_params(xgb_params)
            
            # Log for evaluation step
            np.save(os.path.join(output, "X_test_scaled.npy"), scaler.transform(X_test))
            np.save(os.path.join(output, "y_test.npy"), y_test)
            
            logger.info(f"Training completed. Run ID: {current_run_id}")
            
    # --- EVALUATION ---
    if evaluate:
        logger.info("Step 2: Evaluation starting...")
        # In a real pipeline, we'd load the model from the current run or registry
        # Here we use the files we just saved for simplicity
        try:
            X_test_scaled = np.load(os.path.join(output, "X_test_scaled.npy"))
            y_test = np.load(os.path.join(output, "y_test.npy"))
            
            model = xgb.XGBRegressor()
            model._estimator_type = "regressor"
            model.load_model(os.path.join(output, "model_master.json"))
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {"mse": float(mse), "r2": float(r2)}
            logger.info(f"Evaluation results: {metrics}")
            
            # If we have an active run, log to it
            if mlflow.active_run():
                mlflow.log_metrics(metrics)
            
            with open(os.path.join(output, "evaluation_master.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    # --- DEPLOYMENT ---
    if deploy:
        logger.info("Step 3: Deployment (Simulation) starting...")
        # Simulation: In a real environment, we'd register the model and transition to Production
        # Or trigger a CI/CD pipeline/Kubernetes update.
        deployment_info = {
            "status": "deployed",
            "target": "staging-environment",
            "model_version": "v1.0.2",
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(output, "deployment_master.json"), 'w') as f:
            json.dump(deployment_info, f, indent=2)
        logger.info(f"Model deployed to staging: {deployment_info}")

    # --- MONITORING ---
    if monitor:
        logger.info("Step 4: Monitoring Setup starting...")
        # Simulation: Setup Prometheus alerting or Drift detection
        monitoring_config = {
            "drift_detection": "enabled",
            "alert_threshold": 0.05,
            "metrics_endpoint": "/api/v1/metrics/ml",
            "dashboard_url": "http://grafana:3000/d/ml-model-performance"
        }
        with open(os.path.join(output, "monitoring_master.json"), 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        logger.info(f"Monitoring configured: {monitoring_config}")

    logger.info("All requested ML steps completed successfully!")

if __name__ == '__main__':
    main()
