import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

class AnomalyDetector:
    """
    Unified ML-based anomaly detector for system metrics (latency, error rates, CPU).
    Uses Isolation Forest and StandardScaler for robust outlier detection.
    """

    def __init__(self, contamination: float = 0.05):
        if not (0.0 < contamination < 0.5):
            raise ValueError("Contamination must be between 0 and 0.5.")
        self.model = IsolationForest(contamination=contamination, n_jobs=-1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(self, historical_data: pd.DataFrame | np.ndarray):
        """
        Train the model on historical 'normal' data with feature scaling.
        """
        if isinstance(historical_data, pd.DataFrame):
            if historical_data.empty:
                logger.warning("training_data_empty")
                return
            numeric_df = historical_data.select_dtypes(include=[np.number])
            features = numeric_df.values
            self.columns = list(numeric_df.columns)
        else:
            features = historical_data
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            self.columns = [f"feat_{i}" for i in range(features.shape[1])]

        if features.shape[0] == 0:
            logger.warning("training_data_empty_features")
            return

        # Scale features for better Isolation Forest performance
        scaled_features = self.scaler.fit_transform(features)

        self.model.fit(scaled_features)
        self.is_fitted = True
        logger.info(
            "anomaly_detector_trained",
            samples=len(features),
            features=self.columns,
        )

    def detect(self, current_metrics: pd.DataFrame | np.ndarray) -> list[dict]:
        """
        Detect anomalies in current metrics with optimized scaling and vectorized prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before detection.")

        if isinstance(current_metrics, pd.DataFrame):
            if current_metrics.empty:
                return []
            numeric_df = current_metrics.select_dtypes(include=[np.number])
            features = numeric_df.values
        else:
            features = current_metrics
            if features.ndim == 1:
                features = features.reshape(-1, 1)

        if features.shape[0] == 0:
            return []

        # Scale current features using parameters from training
        scaled_features = self.scaler.transform(features)

        # Vectorized prediction
        predictions = self.model.predict(scaled_features)
        scores = self.model.decision_function(scaled_features)

        anomalies = []
        preds_arr = np.atleast_1d(predictions)
        anomaly_indices = np.where(preds_arr == -1)[0]

        for idx in anomaly_indices:
            anomaly_info = {
                "index": int(idx),
                "score": float(scores[idx]),
            }
            if isinstance(current_metrics, pd.DataFrame):
                anomaly_info["metrics"] = current_metrics.select_dtypes(include=[np.number]).iloc[idx].to_dict()
            else:
                anomaly_info["metrics"] = {f"feat_{i}": float(features[idx, i]) for i in range(features.shape[1])}

            anomalies.append(anomaly_info)
            logger.warning("anomaly_detected", **anomaly_info)

        return anomalies
