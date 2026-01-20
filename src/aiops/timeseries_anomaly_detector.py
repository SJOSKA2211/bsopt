import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import structlog

logger = structlog.get_logger()

class TimeSeriesAnomalyDetector:
    """
    ML-based anomaly detector for system metrics (latency, error rates, CPU).
    Uses Isolation Forest to detect outliers in time-series data.
    """
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False

    def train(self, historical_data: pd.DataFrame):
        """
        Train the model on historical 'normal' data.
        historical_data should be a DataFrame where each row is a time step 
        and columns are different metrics.
        """
        if historical_data.empty:
            logger.warning("training_data_empty")
            return

        # Prepare features (assume all columns are numeric features)
        features = historical_data.select_dtypes(include=[np.number]).values
        
        self.model.fit(features)
        self.is_fitted = True
        logger.info("anomaly_detector_trained", 
                    samples=len(historical_data), 
                    features=list(historical_data.columns))

    def detect(self, current_metrics: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies in the current metrics.
        Returns a list of detected anomalies with their scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before detection.")

        if current_metrics.empty:
            return []

        features = current_metrics.select_dtypes(include=[np.number]).values
        
        # predict() returns -1 for anomalies and 1 for normal points
        predictions = self.model.predict(features)
        scores = self.model.decision_function(features)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                anomaly_info = {
                    "index": i,
                    "score": float(score),
                    "metrics": current_metrics.iloc[i].to_dict()
                }
                anomalies.append(anomaly_info)
                logger.warning("anomaly_detected", **anomaly_info)
        
        return anomalies
