import time
import optuna
import xgboost as xgb
import mlflow
import structlog
from prometheus_client import Summary, Gauge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Callable

# Initialize structured logger
logger = structlog.get_logger()

class InstrumentedTrainer:
    """
    Autonomous model trainer with hyperparameter optimization (Optuna),
    experiment tracking (MLflow), and observability (Prometheus).
    """
    
    # Prometheus metrics
    training_duration_metric = Summary('training_duration_seconds', 'Time spent in training')
    model_accuracy_metric = Gauge('model_accuracy_score', 'Accuracy score of the latest model')

    def __init__(self, study_name: str, storage: str = None):
        self.study_name = study_name
        self.storage = storage
        self.model = None
        self.best_params = {}

    def train_and_evaluate(self, X: Any, y: Any, params: Dict[str, Any]) -> float:
        """
        Trains an XGBoost model and evaluates its performance.
        Logs metrics and parameters to MLflow.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(params)
            
            # Start timer for Prometheus
            start_time = time.time()
            
            # Train model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # XGBoost params (adding some defaults if not in params)
            xgb_params = params.copy()
            n_estimators = xgb_params.pop("n_estimators", 100)
            
            model = xgb.train(xgb_params, dtrain, num_boost_round=n_estimators)
            
            # Predict and evaluate
            y_pred_prob = model.predict(dtest)
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("duration", duration)
            
            # Emit Prometheus metrics
            self.training_duration_metric.observe(duration)
            self.model_accuracy_metric.set(accuracy)
            
            # Update best model if accuracy is better (internal tracking for this session)
            # In a real scenario, you'd probably handle this differently
            self.model = model
            
            logger.info("model_trained", accuracy=accuracy, duration=duration, params=params)
            
            return accuracy

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
