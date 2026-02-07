import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()


class TimeSeriesAnomalyDetector:
    """
    ML-based anomaly detector for system metrics (latency, error rates, CPU).
    Uses Isolation Forest and StandardScaler for robust outlier detection.
    """

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            contamination=contamination, n_jobs=-1, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(self, historical_data: pd.DataFrame):
        """
        Train the model on historical 'normal' data with feature scaling.
        """
        if historical_data.empty:
            logger.warning("training_data_empty")
            return

        # Prepare features (assume all columns are numeric features)
        numeric_df = historical_data.select_dtypes(include=[np.number])
        features = numeric_df.values

        # Scale features for better Isolation Forest performance
        scaled_features = self.scaler.fit_transform(features)

        self.model.fit(scaled_features)
        self.is_fitted = True
        logger.info(
            "anomaly_detector_trained",
            samples=len(historical_data),
            features=list(numeric_df.columns),
        )

    def detect(self, current_metrics: pd.DataFrame) -> list[dict]:
        """
        Detect anomalies in current metrics with optimized scaling and vectorized prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before detection.")

        if current_metrics.empty:
            return []

        numeric_df = current_metrics.select_dtypes(include=[np.number])
        features = numeric_df.values

        # Scale current features using parameters from training
        scaled_features = self.scaler.transform(features)

        # Vectorized prediction
        predictions = self.model.predict(scaled_features)
        scores = self.model.decision_function(scaled_features)

        anomalies = []
        # Optimized loop for finding -1 (anomalies)
        anomaly_indices = np.where(predictions == -1)[0]

        for idx in anomaly_indices:
            anomaly_info = {
                "index": int(idx),
                "score": float(scores[idx]),
                "metrics": numeric_df.iloc[idx].to_dict(),
            }
            anomalies.append(anomaly_info)
            logger.warning("anomaly_detected", **anomaly_info)

        return anomalies
