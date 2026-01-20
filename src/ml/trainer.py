import time
import optuna
import xgboost as xgb
import mlflow
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import tempfile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Callable, Optional, List
from src.shared.observability import (
    TRAINING_DURATION, 
    MODEL_ACCURACY, 
    MODEL_RMSE,
    TRAINING_ERRORS,
    push_metrics
)

# Initialize structured logger
logger = structlog.get_logger()

class BaseTrainer:
    """Base class for all trainers."""
    def train(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def predict(self, model: Any, X: Any) -> Any:
        raise NotImplementedError
    
    def log_model(self, model: Any, artifact_path: str):
        """Logs the model to MLflow."""
        pass

    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Returns feature importance if applicable."""
        return None

class XGBoostTrainer(BaseTrainer):
    def train(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any, params: Dict[str, Any]) -> Any:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        xgb_params = params.copy()
        n_estimators = xgb_params.pop("n_estimators", 100)
        xgb_params.pop("framework", None)
        return xgb.train(xgb_params, dtrain, num_boost_round=n_estimators)

    def predict(self, model: Any, X: Any) -> Any:
        dtest = xgb.DMatrix(X)
        y_pred_prob = model.predict(dtest)
        return (y_pred_prob > 0.5).astype(int)

    def log_model(self, model: Any, artifact_path: str):
        mlflow.xgboost.log_model(model, artifact_path)

    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        importance = model.get_score(importance_type='weight')
        # Map 'f0', 'f1'... to actual names if they are in that format
        result = {}
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            if key in importance:
                result[name] = float(importance[key])
        return result

class SklearnTrainer(BaseTrainer):
    def train(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any, params: Dict[str, Any]) -> Any:
        sk_params = params.copy()
        sk_params.pop("framework", None)
        model = RandomForestClassifier(**sk_params)
        model.fit(X_train, y_train)
        return model

    def predict(self, model: Any, X: Any) -> Any:
        return model.predict(X)

    def log_model(self, model: Any, artifact_path: str):
        mlflow.sklearn.log_model(model, artifact_path)

    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        importances = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}

class PyTorchTrainer(BaseTrainer):
    class SimpleNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.fc(x)

    def train(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any, params: Dict[str, Any]) -> Any:
        epochs = params.get("epochs", 10)
        lr = params.get("lr", 0.01)
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        
        model = self.SimpleNet(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        return model

    def predict(self, model: Any, X: Any) -> Any:
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            outputs = model(X_t)
            return (outputs.numpy() > 0.5).astype(int).flatten()

    def log_model(self, model: Any, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)

class InstrumentedTrainer:
    """
    Autonomous model trainer with hyperparameter optimization (Optuna),
    experiment tracking (MLflow), and observability (Prometheus).
    """
    
    def __init__(self, study_name: str, storage: str = None, tracking_uri: str = None):
        self.study_name = study_name
        self.storage = storage
        self.model = None
        self.best_params = {}
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.trainers = {
            "xgboost": XGBoostTrainer(),
            "sklearn": SklearnTrainer(),
            "pytorch": PyTorchTrainer()
        }

    def push_metrics(self):
        """Push metrics to Prometheus Gateway."""
        push_metrics(job_name=self.study_name)

    def _plot_feature_importance(self, importance: Dict[str, float], framework: str) -> str:
        """Creates a feature importance plot and returns the path to the saved image."""
        plt.figure(figsize=(10, 6))
        names = list(importance.keys())
        values = list(importance.values())
        plt.barh(names, values)
        plt.title(f"Feature Importance ({framework})")
        plt.xlabel("Importance")
        
        temp_dir = tempfile.mkdtemp()
        plot_path = os.path.join(temp_dir, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def train_and_evaluate(self, X: Any, y: Any, params: Dict[str, Any], feature_names: List[str] = None, dataset_metadata: Optional[Dict[str, str]] = None) -> float:
        """
        Trains a model and evaluates its performance.
        Logs metrics, parameters, and artifacts to MLflow.
        """
        framework = params.get("framework", "xgboost")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            trainer = self.trainers.get(framework, self.trainers["xgboost"])
            
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                
                if dataset_metadata:
                    mlflow.set_tags(dataset_metadata)
                
                start_time = time.time()
                model = trainer.train(X_train, y_train, X_test, y_test, params)
                y_pred = trainer.predict(model, X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                duration = time.time() - start_time
                
                # Dummy RMSE for classification (1.0 - accuracy) for demo purposes
                rmse = 1.0 - accuracy
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("duration", duration)
                
                # Log model and artifacts
                trainer.log_model(model, "model")
                
                importance = trainer.get_feature_importance(model, feature_names)
                if importance:
                    plot_path = self._plot_feature_importance(importance, framework)
                    mlflow.log_artifact(plot_path)
                    # Cleanup temp plot
                    os.remove(plot_path)
                    os.rmdir(os.path.dirname(plot_path))
                
                TRAINING_DURATION.labels(framework=framework).observe(duration)
                MODEL_ACCURACY.labels(framework=framework).set(accuracy)
                MODEL_RMSE.labels(model_type=framework, dataset="validation").set(rmse)
                
                self.model = model
                logger.info("model_trained", framework=framework, accuracy=accuracy, rmse=rmse, duration=duration, params=params)
                
                return float(accuracy)
        except Exception as e:
            TRAINING_ERRORS.labels(framework=framework).inc()
            logger.error("training_failed", framework=framework, error=str(e))
            raise
        finally:
            self.push_metrics()

    def optimize(self, objective: Callable, n_trials: int = 20) -> optuna.study.Study:
        """
        Runs the hyperparameter optimization loop.
        """
        logger.info("optimization_started", study_name=self.study_name, n_trials=n_trials)
        
        study = optuna.create_study(study_name=self.study_name, direction="maximize", storage=self.storage, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info("optimization_completed", best_params=self.best_params, best_value=study.best_value)
        
        return study
